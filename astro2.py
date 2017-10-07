import numpy as np
import matplotlib.pyplot as plt
from numpy import unravel_index 
from astropy.visualization import astropy_mpl_style
from astropy.io import fits
from scipy.odr import *
from matplotlib import mlab

class Image:
    def __init__(self):
        hdulist=fits.open('mosaic.fits')
        hdulist.info()
        self.header =  hdulist[0].header
        self.image_data = hdulist[0].data
        #NB: cut_edges only to be called at the initialisation 
        self.cut_edges()
        self.mask_bright()
        self.catalogue = {}
        hdulist.close()
    
    def original_image(self, v_min = 3300, v_max = 3800):
        hdulist=fits.open('mosaic.fits')
        data = hdulist[0].data
        plt.style.use(astropy_mpl_style)
        plt.imshow(data, cmap='gray', vmin = v_min, vmax=v_max)
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.grid(False)
        plt.gca().invert_yaxis()
        plt.colorbar()
        
    def visualise(self, v_min = 3300, v_max = 3800):
        data = self.image_data
        plt.style.use(astropy_mpl_style)
        plt.imshow(data, cmap='gray', vmin=v_min, vmax=v_max)
        plt.gca().invert_yaxis()
        plt.grid(False)
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.colorbar()

    def cut_edges(self):
        #self.image_data = self.image_data[115:4518,115:2478]     
        self.image_data[0:4611, 0:230] = np.full((4611, 230), 1)
        self.image_data[0:230, 0:2570] = np.full((230, 2570), 1)
        self.image_data[0:4611, 2340:2570] = np.full((4611, 230), 1)
        self.image_data[4381:4611,  0:2570] = np.full((230, 2570), 1)
    
    def find_coord_x(self, x, x_size):
        return int(x - (x_size/2.)), int(x + (x_size/2.) +1)
        
    def find_coord_y(self, y, y_size):
        return int(y - (y_size/2.)), int(y + (y_size/2.) +1)

    def mask_bright(self):
        #remember to swap x and y   
        self.image_data[2949:3474, 1150:1695] = np.full((525, 545), 1)     
        x_min, x_max = self.find_coord_x(130,77)
        y_min, y_max = self.find_coord_y(1516,52) 
        self.image_data[x_min:x_max, y_min:y_max] = np.full((x_max-x_min, y_max-y_min), 1)
        
        x_min, x_max = self.find_coord_x(126,42)
        y_min, y_max = self.find_coord_y(1337,125) 
        self.image_data[x_min:x_max, y_min:y_max] = np.full((x_max-x_min, y_max-y_min), 1)
        
        x_min, x_max = self.find_coord_x(202,201)
        y_min, y_max = self.find_coord_y(1441,97)
        self.image_data[x_min:x_max, y_min:y_max] = np.full((x_max-x_min, y_max-y_min), 1)
        
        x_min, x_max = self.find_coord_x(429,39)
        y_min, y_max = self.find_coord_y(1569,159)
        self.image_data[x_min:x_max, y_min:y_max] = np.full((x_max-x_min, y_max-y_min), 1)
       
        x_min, x_max = self.find_coord_x(446,59)
        y_min, y_max = self.find_coord_y(1343,301)
        self.image_data[x_min:x_max, y_min:y_max] = np.full((x_max-x_min, y_max-y_min), 1)
       
        x_min, x_max = self.find_coord_x(368,111)
        y_min, y_max = self.find_coord_y(1438,49)
        self.image_data[x_min:x_max, y_min:y_max] = np.full((x_max-x_min, y_max-y_min), 1)
        
        x_min, x_max = self.find_coord_x(334,42)
        y_min, y_max = self.find_coord_y(1429,536)
        self.image_data[x_min:x_max, y_min:y_max] = np.full((x_max-x_min, y_max-y_min), 1)
        
        x_min, x_max = self.find_coord_x(3211,524)
        y_min, y_max = self.find_coord_y(1422,544)
        self.image_data[x_min:x_max, y_min:y_max] = np.full((x_max-x_min, y_max-y_min), 1)
        
        x_min, x_max = self.find_coord_x(1428,73)
        y_min, y_max = self.find_coord_y(2089,73)
        self.image_data[x_min:x_max, y_min:y_max] = np.full((x_max-x_min, y_max-y_min), 1)
        
        x_min, x_max = self.find_coord_x(1777,90)
        y_min, y_max = self.find_coord_y(1420,94)
        self.image_data[x_min:x_max, y_min:y_max] = np.full((x_max-x_min, y_max-y_min), 1)
       
        x_min, x_max = self.find_coord_x(2525,4159)
        y_min, y_max = self.find_coord_y(1437,28)
        self.image_data[x_min:x_max, y_min:y_max] = np.full((x_max-x_min, y_max-y_min), 1)
        
        x_min, x_max = self.find_coord_x(3762,111)
        y_min, y_max = self.find_coord_y(2142,87)
        self.image_data[x_min:x_max, y_min:y_max] = np.full((x_max-x_min, y_max-y_min), 1)
        
        x_min, x_max = self.find_coord_x(4096,66)
        y_min, y_max = self.find_coord_y(563,73)
        self.image_data[x_min:x_max, y_min:y_max] = np.full((x_max-x_min, y_max-y_min), 1)
        
        x_min, x_max = self.find_coord_x(1488,63)
        y_min, y_max = self.find_coord_y(639,70)
        self.image_data[x_min:x_max, y_min:y_max] = np.full((x_max-x_min, y_max-y_min), 1)
        
        x_min, x_max = self.find_coord_x(2290,149)
        y_min, y_max = self.find_coord_y(907,125)
        self.image_data[x_min:x_max, y_min:y_max] = np.full((x_max-x_min, y_max-y_min), 1)
        
        x_min, x_max = self.find_coord_x(3413,70)
        y_min, y_max = self.find_coord_y(2469,70)
        self.image_data[x_min:x_max, y_min:y_max] = np.full((x_max-x_min, y_max-y_min), 1)
        
        x_min, x_max = self.find_coord_x(3311,217)
        y_min, y_max = self.find_coord_y(779,130)
        self.image_data[x_min:x_max, y_min:y_max] = np.full((x_max-x_min, y_max-y_min), 1)
        
        x_min, x_max = self.find_coord_x(2767,128)
        y_min, y_max = self.find_coord_y(978,115)
        self.image_data[x_min:x_max, y_min:y_max] = np.full((x_max-x_min, y_max-y_min), 1)
        
        x_min, x_max = self.find_coord_x(2304,70)
        y_min, y_max = self.find_coord_y(453,70)
        self.image_data[x_min:x_max, y_min:y_max] = np.full((x_max-x_min, y_max-y_min), 1)
        
        x_min, x_max = self.find_coord_x(2310, 63)
        y_min, y_max = self.find_coord_y(2133, 87)
        self.image_data[x_min:x_max, y_min:y_max] = np.full((x_max-x_min, y_max-y_min), 1)
       
    def background_hist(self):
        #data = self.image_data[115:4518,115:2478]
        data = self.image_data        
        data = data.flatten()
        data_to_plot = []
        for count in data:
            if 3000 < count < 4000:
                data_to_plot.append(count)
        x = np.linspace(3350, 3500, 1000)
        plt.plot(x, mlab.normpdf(x, 3419, 12), 'r', label = '$\mu = 3419$  $\sigma = 12$')
        plt.hist(data_to_plot, bins = 300, range = (3300, 3600), normed = True, fill = False)
        plt.xlabel('$Count$ $per$ $Pixel$')
        plt.ylabel('$Probability$ $Density$')
        plt.legend()
        plt.show()
    
    def find_ring(self, x0, y0, radius):
        radius_2 = radius + 1
        ring = []
        range_x2 = range(x0-radius_2,x0+radius_2+1)
        range_y2 = range(y0-radius_2, y0+radius_2+1)
        for x in range_x2:
            for y in range_y2:
                if ((x - x0)**2 + (y - y0)**2) <= radius_2**2:
                    if ((x - x0)**2 + (y - y0)**2) > radius**2:
                        ring.append(self.image_data[x][y])
        return ring
        
    def back_ring(self, x0, y0, radius):
        radius_2 = radius + 20
        ring = []
        range_x2 = range(x0-radius_2,x0+radius_2+1)
        range_y2 = range(y0-radius_2, y0+radius_2+1)
        for x in range_x2:
            for y in range_y2:
                if ((x - x0)**2 + (y - y0)**2) <= radius_2**2:
                    if ((x - x0)**2 + (y - y0)**2) > radius**2:
                        if self.image_data[x, y] < (3419 + (4*12)):
                            ring.append(self.image_data[x][y])
        return ring
        
    def source_detection(self):
        '''Return the number of galaxies counted and produce dictionary with
        keys corresponding to physical coordinates and values containing the 
        radius, the pixels count and the local background of the source detected.
        '''
        data = self.image_data
        for i in range(5000):
            x0, y0 = unravel_index(data.argmax(), data.shape)
            #changed this from 5 to 6
            if self.image_data[x0, y0] >= (3419+(5*12)): 
                radius = 3
                ring = self.find_ring(x0, y0, radius)
            
                while np.mean(ring) >= (3419 + (2*12)):
                    radius = radius + 1
                    ring = self.find_ring(x0, y0, radius)
                
                if np.mean(ring) < (3419 + (2*12)):
                    back_ring = self.back_ring(x0, y0, radius)
 
                range_x = range(x0-radius,x0+radius+1)
                range_y = range(y0-radius, y0+radius+1)
                pixels = 0
                count = 0
                for x in range_x:
                    for y in range_y:                        
                        if ((x - x0)**2 + (y - y0)**2) < radius**2:
                            pixels = pixels + 1
                            count = count + self.image_data[x, y]
                            self.image_data[x, y] = 3419
                        if ((x - x0)**2 + (y - y0)**2) == radius**2:
                            #to account for elliptical source
                            if self.image_data[x, y] > (3419 + (4*12)):
                                pixels = pixels + 1
                                count = count + self.image_data[x, y]
                                self.image_data[x, y] = 3419    

                local_back = np.mean(back_ring)*pixels                   
                flux = count - local_back 
                self.catalogue[(x0, y0)] = [radius, flux, local_back]
            else:
                return i
                
    def magnitudes(self):
        magnitude = []
        for source in self.catalogue.keys():
            mag = 2.530E01 + -2.5*np.log10(self.catalogue[source][1])
            magnitude.append(mag)
        return magnitude
    
    def N_vs_m(self, first, last):
        N = []
        m = []
        for i in np.arange(min(self.magnitudes())+2, max(self.magnitudes())+1, 0.5):
            m.append(i)
            n = 0
            for mag in self.magnitudes():
                if mag < i:
                    n = n + 1
            N.append(n)
        #plt.plot(m, np.log10(N), '.')
        #plotting linear fitting
        def linear_func(p, x):
             m, c = p
             return m*x + c
        
        N = np.array(N)/0.06
        x= np.array(m)
        y= np.log10(N)
        #y = np.log10(N)/((0.6*x)-2.683)
        y_error= 1./(np.sqrt(np.array(N)))
        y_error_up= np.log10(N) - np.log10(N-np.sqrt(N))
        y_error_down = np.log10(N+np.sqrt(N)) - np.log10(N)
        
        # Create a model for fitting.
        linear_model = Model(linear_func)
        # Create a RealData object using our initiated data from above.
        data = RealData(x[first:-last], y[first:-last], sy= y_error[first:-last])
        # Set up ODR with the model and data.
        odr = ODR(data, linear_model, beta0=[0., 0.])
        # Run the regression.
        out = odr.run()
      
        x_fit = np.linspace(x[first], x[-last], 1000)
        y_fit = linear_func(out.beta, x_fit)
        plt.errorbar(x, y, yerr=[y_error_up, y_error_down], linestyle='None', marker='x', color = 'black')
        plt.plot(x_fit, y_fit)
       
        plt.show()
        plt.xlabel('$m$')
        plt.ylabel('$log(N(m))$')
        plt.grid(False)
        return out.pprint()
    
    def plot_source(self, v_min, v_max):
        source = self.catalogue
        hdulist=fits.open('mosaic.fits')
        data = hdulist[0].data
        plt.imshow(data, cmap='gray', vmin=v_min, vmax=v_max)   
        plt.gca().invert_yaxis()
        plt.colorbar()
        for i in range(len(source.items())):
            circle_x = []
            circle_y = []
            x = source.keys()[i][0]
            y = source.keys()[i][1]
            radius = source.values()[i][0]
            plt.style.use(astropy_mpl_style)
            if x < 2000:
                if x >1000:
                    if y > 1500:
                        if y < 2000:
                            for x_coor in np.arange(x-2*radius, x+2*radius, 0.1):
                                for y_coor in np.arange(y-radius-0.5, y+radius+0.5, 0.1):
                                    if ((x_coor - x)**2 + (y_coor - y)**2) > (radius-0.005)**2:
                                        if ((x_coor - x)**2 + (y_coor - y)**2) < (radius+0.005)**2:
                                            circle_x.append(x_coor)
                                            circle_y.append(y_coor)
            plt.plot(circle_y, circle_x,'r.', markersize = 1)
            plt.grid(False)
            plt.xlabel('$x$')
            plt.ylabel('$y$')
    
    def plot_aperture(self, galaxy, v_min = 3300, v_max = 3800):
        source = self.catalogue
        hdulist=fits.open('mosaic.fits')
        data = hdulist[0].data
        plt.imshow(data, cmap='gray', vmin=v_min, vmax=v_max)   
        plt.gca().invert_yaxis()
        plt.colorbar()
        circle_x = []
        circle_y = []
        annulus_x = []
        annulus_y = []
        x = source.keys()[galaxy][0]
        y = source.keys()[galaxy][1]
        radius = source.values()[galaxy][0]
        plt.style.use(astropy_mpl_style)
        for x_coor in np.arange(x-2*radius, x+2*radius, 0.1):
            for y_coor in np.arange(y-radius-0.5, y+radius+0.5, 0.1):
                if ((x_coor - x)**2 + (y_coor - y)**2) > (radius-0.005)**2:
                    if ((x_coor - x)**2 + (y_coor - y)**2) < (radius+0.005)**2:
                        circle_x.append(x_coor)
                        circle_y.append(y_coor)
        for x_coor in np.arange(x-2*(radius+20), x+2*(radius+20), 0.1):
            for y_coor in np.arange(y-(radius+20)-0.5, y+(radius+20)+0.5, 0.1):
                if ((x_coor - x)**2 + (y_coor - y)**2) > (radius+20-0.005)**2:
                    if ((x_coor - x)**2 + (y_coor - y)**2) < (radius+20+0.005)**2:
                        annulus_x.append(x_coor)
                        annulus_y.append(y_coor)
        plt.plot(circle_y, circle_x,'r.', markersize = 3, label = 'Circular Aperture')
        plt.plot(annulus_y, annulus_x,'g.', markersize = 3, label = 'Annulus')
        plt.legend()
        plt.grid(False)
        plt.xlabel('$x$')
        plt.ylabel('$y$')
    
def run():
    im = Image()
    print im.source_detection()
    im.N_vs_m()