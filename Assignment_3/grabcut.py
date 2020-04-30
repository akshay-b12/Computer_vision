import numpy as np
import cv2
import glob
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal as mvn
import igraph
from copy import deepcopy
import matplotlib.pyplot as plt

"""### GMM implementation"""

from sklearn.cluster import KMeans
import numpy as np

class GaussianMixture:
    def __init__(self, X, n_components=5):
        self.n_components = n_components
        self.n_features = X.shape[1]
        self.n_samples = np.zeros(self.n_components)
        
        ## pi- mixing coefficient matrix
        self.coefs = np.zeros(self.n_components)
        ## means of n_components gaussains
        self.means = np.zeros((self.n_components, self.n_features))
        ## covariances matrices of n_component gaussians
        self.covariances = np.zeros((self.n_components, self.n_features,
                                     self.n_features))
        #print(X.shape)
        ## 
        self.init_with_kmeans(X)

    def init_with_kmeans(self, X):
        ## initialize KMeans with n_components and get labels of each point
        label = KMeans(n_clusters=self.n_components, n_init=1).fit(X).labels_
        ## Compute k-means clustering
        self.fit(X, label)

    def calc_score(self, X, ci):
        score = np.zeros(X.shape[0])
        if self.coefs[ci] > 0:
            diff = X - self.means[ci]
            mult = np.einsum(
                'ij,ij->i', diff,
                np.dot(np.linalg.inv(self.covariances[ci]), diff.T).T)
            score = np.exp(-.5 * mult) / np.sqrt(2 * np.pi) / \
                np.sqrt(np.linalg.det(self.covariances[ci]))

        return score

    def calc_prob(self, X):
        prob = [self.calc_score(X, ci) for ci in range(self.n_components)]
        return np.dot(self.coefs, prob)

    def which_component(self, X):
        prob = np.array(
            [self.calc_score(X, ci) for ci in range(self.n_components)]).T
        return np.argmax(prob, axis=1)

    def fit(self, X, labels):
        assert self.n_features == X.shape[1]

        self.n_samples[:] = 0
        self.coefs[:] = 0

        uni_labels, count = np.unique(labels, return_counts=True)
        self.n_samples[uni_labels] = count

        variance = 0.01
        for ci in uni_labels:
            n = self.n_samples[ci]

            self.coefs[ci] = n / np.sum(self.n_samples)
            self.means[ci] = np.mean(X[ci == labels], axis=0)
            self.covariances[ci] = 0 if self.n_samples[ci] <= 1 else np.cov(
                X[ci == labels].T)

            det = np.linalg.det(self.covariances[ci])
            if det <= 0:
                self.covariances[ci] += np.eye(self.n_features) * variance
                det = np.linalg.det(self.covariances[ci])

### GrabCut ###

BG = 0
FG = 1
PR_BG = 2
PR_FG = 3

class Grabcut:
    def __init__ (self, ip_img, mask, rect, num_gmm = 5, gamma = 50, neighbors = 8):
        self.img = np.asarray(ip_img, dtype=np.float64)
        self.out_image = deepcopy(ip_img)
        self.rows, self.cols, _ = self.img.shape
        self.mask = mask
        if rect is not None:
            self.mask[rect[1]:rect[1] + rect[3],rect[0]:rect[0] + rect[2]] = PR_FG
        self.assign_fg_bg()
        self.gmm_comp = num_gmm
        self.gamma = gamma
        self.neighbours = neighbors
        self.calc_smoothness_term()
        
        ## Initialize fg and bg GMMs with th respective pixels as data points 
        self.fg_gmm = GaussianMixture(self.img[self.fgd_indexes], n_components = self.gmm_comp)
        self.bg_gmm = GaussianMixture(self.img[self.bgd_indexes], n_components = self.gmm_comp)
        
        ## GMM label of each pixel
        self.label_gmm = np.empty((self.rows, self.cols), dtype=np.uint32)
        
        ## Define two extra nodes of source and sink with there node numbers as n*m, n*m+1
        self.source_s = self.cols * self.rows
        self.sink_t = self.source_s + 1
        
    def assign_fg_bg(self):
        self.bgd_indexes = np.where(np.logical_or(self.mask == BG, self.mask == PR_BG))
        #print(self.fg_idx)
        self.fgd_indexes = np.where(np.logical_or(self.mask == FG, self.mask == PR_FG))
        
    def calc_smoothness_term(self):
        left_pix_diff = self.img[:, 1:] - self.img[:,:-1]
        upleft_pix_diff = self.img[1:, 1:] - self.img[:-1, :-1]
        up_pix_diff = self.img[1:, :] - self.img[:-1, :]
        upright_pix_diff = self.img[1:, :-1] - self.img[:-1, 1:]
        
        #self.beta = np.linalg.norm(left_pix_diff)**2 + np.linalg.norm(upleft_pix_diff)**2 + np.linalg.norm(up_pix_diff)**2 + np.linalg.norm(upright_pix_diff)**2
        
        self.beta = np.linalg.norm(left_pix_diff)**2 + np.linalg.norm(up_pix_diff)**2
        if self.neighbours == 8:
            self.beta += np.linalg.norm(upleft_pix_diff)**2 + np.linalg.norm(
                upright_pix_diff)**2
        #         self.beta = np.sum(np.square(left_diff)) + np.sum(np.square(up_left_diff)) + \
        #             np.sum(np.square(up_diff)) + \
        #             np.sum(np.square(up_right_diff))
        
        ## Here denominator denotes number of operations(subtractions) performed in finding expectation E
        if self.neighbours == 8:
            self.beta = 2.0 * self.beta / (4 * self.rows * self.cols - 3 *
                                           (self.rows + self.cols) + 2.0)
        else:
            self.beta = 2.0 * self.beta / (2 * self.rows * self.cols - 1 *
                                           (self.rows + self.cols) + 2.0)
        self.beta = 1 / self.beta

        #print(self.beta)
        ## gamma = 50, Find smootness term individually 
        self.left_V = self.gamma * np.exp(-self.beta * np.sum(np.square(left_pix_diff), axis=2))
        
        self.upleft_V = self.gamma / np.sqrt(2) * np.exp(-self.beta * np.sum(np.square(upleft_pix_diff), axis=2))
        
        self.up_V = self.gamma * np.exp(-self.beta * np.sum(np.square(up_pix_diff), axis=2))
        
        self.upright_V = self.gamma / np.sqrt(2) * np.exp(-self.beta * np.sum(np.square(upright_pix_diff), axis=2))
        
    def assign_GMM(self):
        ## Assign a gaussain out of n_components gaussians to each fg pix
        ## depending its argmax y=N(mu,sigma,x) value
        self.label_gmm[self.fgd_indexes] = self.fg_gmm.which_component(
            self.img[self.fgd_indexes])
        ## Assign a gaussain out of n_components gaussians to each bg pix
        ## depending its argmax y=N(mu,sigma,x) value
        self.label_gmm[self.bgd_indexes] = self.bg_gmm.which_component(
            self.img[self.bgd_indexes])

    def learn_GMM(self):
        ## Deoending on Xi's assigned to each fg gmm, update its pi,mean, cov matrix 
        self.fg_gmm.fit(self.img[self.fgd_indexes],
                         self.label_gmm[self.fgd_indexes])
        ## Deoending on Xi's assigned to each bg gmm, update its pi,mean, cov matrix 
        self.bg_gmm.fit(self.img[self.bgd_indexes],
                         self.label_gmm[self.bgd_indexes])

    def make_n_links(self, mask1, mask2, V):
        mask1 = mask1.reshape(-1)
        mask2 = mask2.reshape(-1)
        self.gc_graph_capacity += V.reshape(-1).tolist()
        return list(zip(mask1, mask2))

    def construct_gc_graph(self):
        ## Get indices (between [0,m*n], which corresponds to node number in the graph)
        ## of fg, bg and probable fg/bg
        fgd_indexes = np.where(self.mask.reshape(-1) == FG)
        bgd_indexes = np.where(self.mask.reshape(-1) == BG)
        pr_indexes = np.where((self.mask.reshape(-1) == PR_FG)
                              | (self.mask.reshape(-1) == PR_BG))

        self.gc_graph_capacity = []
        edges = []
        
        ## Function creates edges between a given node 'source' and set of nodes 'sink'
        def make_edges(source, sinks):
            return list(zip([source] * sinks[0].size, sinks[0]))
        
        ## Create edges b/w source_s and nodes(pixels) which are probable fg/bg
        edges += make_edges(self.source_s, pr_indexes)
        
        self.gc_graph_capacity += list(-np.log(
            self.bg_gmm.calc_prob(self.img.reshape(-1, 3)[pr_indexes])))
        
        ## Create edges b/w sink_t and nodes(pixels) which are probable fg/bg
        edges += make_edges(self.sink_t, pr_indexes)
        ## Find the data term(U) for 'probable fg/bg' nodes as -log(p(.)) 
        ## where p(.) is gaussian probability distribution
        self.gc_graph_capacity += list(-np.log(
            self.fg_gmm.calc_prob(self.img.reshape(-1, 3)[pr_indexes])))
        
        ## Create edges b/w source_s and fg nodes(pixels)
        edges += make_edges(self.source_s, fgd_indexes)
        self.gc_graph_capacity += [9 * self.gamma] * fgd_indexes[0].size

        ## Create edges b/w sink_t and fg nodes(pixels)
        edges += make_edges(self.sink_t, fgd_indexes)
        self.gc_graph_capacity += [0] * fgd_indexes[0].size
        
        ## Create edges b/w source_s and bg nodes(pixels)
        edges += make_edges(self.source_s, bgd_indexes)
        self.gc_graph_capacity += [0] * bgd_indexes[0].size
        
        ## Create edges b/w sink_t and fg nodes(pixels)
        edges += make_edges(self.sink_t, bgd_indexes)
        self.gc_graph_capacity += [9 * self.gamma] * bgd_indexes[0].size

        
        img_indexes = np.arange(
            self.rows * self.cols, dtype=np.uint32).reshape(
                self.rows, self.cols)
        
        ## Give edge weights to links which are with left node
        edges += self.make_n_links(img_indexes[:, 1:], img_indexes[:, :-1],
                                   self.left_V)
        ## Give edge weights to links which are with upper-left node
        edges += self.make_n_links(img_indexes[1:, 1:], img_indexes[:-1, :-1],
                                   self.upleft_V)
        ## Give edge weights to links which are with the upper nodes(pixels)
        edges += self.make_n_links(img_indexes[1:, :], img_indexes[:-1, :],
                                   self.up_V)
        ## Give edge weights to links which are with upper-left node(pix)
        edges += self.make_n_links(img_indexes[1:, :-1], img_indexes[:-1, 1:],
                                   self.upright_V)

        assert (len(edges) == len(self.gc_graph_capacity))
        
        ## Initialize igraph object with m*n+2 vertices
        self.gc_graph = igraph.Graph(self.rows * self.cols + 2)
        ## Add edges to the graph passed as tuple(u,v) that are formed above
        self.gc_graph.add_edges(edges)

    def estimate_segmentation(self):
        ## Perform st-mincut on the graph
        mincut = self.gc_graph.st_mincut(self.source_s, self.sink_t,
                                         self.gc_graph_capacity)
        print('foreground pixels: %d, background pixels: %d' % (len(
            mincut.partition[0]), len(mincut.partition[1])))
        ## Find the indices of the probable fg/bg from the mask
        pr_indexes = np.where((self.mask == PR_FG) | (self.mask == PR_BG))
        ## Get vertex numbers from 0 to m*n
        img_indexes = np.arange(
            self.rows * self.cols, dtype=np.uint32).reshape(
                self.rows, self.cols)
        
        ## Get the vertices which are part of 1st partition,
        ## 'condition' is True for vertices which are present in 'mincut.partition[0]'
        condition = np.isin(img_indexes[pr_indexes], mincut.partition[0])
        ## Store the values as 'PR_FG' where 'condition' is True 
        ## and 'PR_BG' at indices where the 'condition' is False
        ## i.e mark the probable (unknown) fg and bg pixels at this iteration
        self.mask[pr_indexes] = np.where(condition, PR_FG, PR_BG)
        ## Assign to pixels, there respective updated category
        self.assign_fg_bg()

    def calculate_energy(self):
        U = 0
        bg_indexes = np.where((self.mask == BG) | (self.mask == PR_BG))
        fg_indexes = np.where((self.mask == FG) | (self.mask == PR_FG))
        for component in range(self.gmm_components):
            indexes = np.where((self.label_gmm == component) & (fg_indexes))
            U += np.sum(-np.log(self.fg_gmm.coefs[component] * self.fg_gmm.
                                calc_score(self.img[indexes], component)))
            indexes = np.where((self.label_gmm == component) & (bg_indexes))
            U += np.sum(-np.log(self.bg_gmm.coefs[component] * self.bg_gmm.
                                calc_score(self.img[indexes], component)))

        V = 0
        mask = self.mask.copy()
        mask[bg_indexes] = BG
        mask[fg_indexes] = FG

        V += np.sum(self.left_V * (mask[:, 1:] == mask[:, :-1]))
        V += np.sum(self.upleft_V * (mask[1:, 1:] == mask[:-1, :-1]))
        V += np.sum(self.up_V * (mask[1:, :] == mask[:-1, :]))
        V += np.sum(self.upright_V * (mask[1:, :-1] == mask[:-1, 1:]))
        return U, V, U + V

    def modified_image(self):
        img2 = deepcopy(self.out_image)
        mask = self.mask.copy()
        mask2 = np.where((self.mask == 1) + (self.mask == 3), 255,
                         0).astype('uint8')
        #print(type(img2))
        #print(type(mask2))
        return cv2.bitwise_and(img2, img2, mask=mask2)
        
    def run(self, num_iters=2):
        for i in range(num_iters):
            ## Find argmax (y=N(mu,sig,x)), of each data point (pix) for both fg and bg pix 
            ## and  assign a Gaussain to each X depending on max y
            self.assign_GMM()
            ## Depending on assignments of Xi's to each gmmof fg and bg, update its pi, mean, cov
            self.learn_GMM()
            ## Construct graph: Create and connect nodes source, sink, fg, bg, and give them weights            
            self.construct_gc_graph()
            ## Peform st-mincut on the graph created above and 
            ## update the labels of the pixels as FG, BG, or PR_FG, PR_BG in this iteration
            self.estimate_segmentation()
            
            ########### Code to see effect of number of iterations ##########
            #print("Output after iteration ",i,":")
            #output = self.modified_image()
            #plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
            #plt.show()

drawing = False # true if mouse is pressed
ix,iy = -1,-1
jx, jy = -1, -1
def draw_rect(event,x,y,flags,param):
    global ix, iy, jx, jy, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
            jx, jy = x, y
            #return [(ix,iy), (x,y)]
            
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)

path = input("Enter image path :")
input_img = cv2.imread(path)
img = deepcopy(input_img)
print("Draw box around the object using mouse and then press 'c' to continue")
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_rect)
while(1) :
    cv2.imshow('image',img)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('c'):
        #print("Press 'c' to continue after marking foreground")
        #print(coord)
        print("Top left corner coord: ",ix,iy,"Bottom right corner coor: ", jx,jy)
        p,q,r,s = int(ix), int(iy), int(jx), int(jy)
        #rect = (q, p, abs(q-s), abs(p-r))  # (x,y,width, height) of rect
        rect = (p, q, abs(p-r), abs(q-s))  # (x,y,width, height) of rect
        #print(p,q,r,s)
        mask = np.zeros(input_img.shape[:2], dtype=np.uint8)
        #num_iterations = input("Enter number_of_iterations: ")
        num_iterations = 3
        gc_obj = Grabcut(input_img, mask, rect)
        gc_obj.run(int(num_iterations))
        output = gc_obj.modified_image()
        plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
        plt.show()
        #cv2.imshow('output', output)
        #k = cv2.waitKey()
        cv2.destroyAllWindows()
        ans = input("Want to continue (y/n)?")
        if ans == 'y' or ans == 'Y':
            path = input("Enter image path :")
            input_img = cv2.imread(path)
            img = deepcopy(input_img)
            print("Draw box around the object using mouse and then press 'c' to continue")
            cv2.namedWindow('image')
            cv2.setMouseCallback('image',draw_rect)
            continue
        else :
            break
