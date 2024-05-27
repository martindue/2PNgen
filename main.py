import numpy as np
import argparse
import matplotlib.pyplot as plt
import os,  sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from TwoParamNoiseGen.genNoise import genNoise

def main():

    #TODO: Implement CLI for N, RMS_STDs, Magnitudes, AR, and rotangle
    parser = argparse.ArgumentParser(description='Generate and plot noise data.')
    parser.add_argument('-N', type=int, default=200, help='Number of data points')
    parser.add_argument('-RMS_STDs', nargs='+', type=float, default=[0.1, 0.4, 0.7, 1., 1.3, 1.6, 1.9], help='List of RMS_STD values')
    parser.add_argument('-magnitudes', nargs='+', type=float, default=[0.2, 0.4, 0.6, 0.8, 1.0], help='List of magnitudes')
    parser.add_argument('-AR', type=int, default=1, help='Anisotropic ratio')
    parser.add_argument('-rotangle', type=int, default=0, help='Rotation angle')
    parser.add_argument('-o', '--output', type=str, default='output.png', help='Output file name')
    parser.add_argument('-genFun', type=str, default='uniform', help='Generator function')
    args = parser.parse_args()

    N = args.N
    RMS_STDs = args.RMS_STDs
    magnitudes = args.magnitudes
    AR = args.AR
    rotAng = args.rotangle
    output = args.output
    genFunStr = args.genFun



    if genFunStr == 'uniform':
        genFun = lambda x: np.random.rand(1,x)
        lbl = 'uniform'
    elif genFunStr == 'gaussian':
        genFun = lambda x: np.random.randn(1,x)
        lbl = 'Gaussian'


    generatedNoise = {}
    for i, magnitude in enumerate(magnitudes):
        for j, rms_std in enumerate(RMS_STDs):
            # Generate the data
            print('Generating noise for RMS_STD = ' + str(rms_std) + ' and magnitude = ' + str(magnitude))
            generatedNoise[(i,j)] =  genNoise(N,lbl,genFun, rms_std, magnitude, AR, rotAng)


    print('Data generated. Plotting...')
    # further processing before plotting
    sFac = []
    for key in generatedNoise.keys():
        x = generatedNoise[key]
        sFac.append( max(np.ptp(x[0,:]), np.ptp(x[1,:])))
    max_sFac = max(sFac)

    for key in generatedNoise.keys():
        generatedNoise[key] = generatedNoise[key] / max_sFac

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.subplots_adjust(left=0.2, right=0.8, top=0.9, bottom=0.1)
    voffs = [1, 0.45, 0.55, 0.85, 0.95]

    for key in generatedNoise.keys():
        data = generatedNoise[key]
        x_data = data[0,:] + key[1]  
        y_data = data[1,:] + key[0]

        #add the first element to the end of the array to close the loop
        x_data = np.concatenate((x_data, [x_data[0]]))
        y_data = np.concatenate((y_data, [y_data[0]]))

        ax.plot(x_data, y_data, 'k')



    ax.set_aspect('equal', 'box')
    ax.set_xticks(range(0, len(RMS_STDs) ))
    ax.set_xticklabels([f'{x:.2f}' for x in RMS_STDs])
    ax.set_yticks(range(0, len(magnitudes) ))
    ax.set_yticklabels([f'{x:.1f}' for x in magnitudes])
    ax.set_xlabel(r'$\frac{\mbox{RMS-S2S}}{\mbox{STD}}$', fontsize=12, usetex=True)
    ax.set_ylabel(r'$\sqrt{\mbox{RMS-S2S}^2+\mbox{STD}^2}$ ($^\circ$)', fontsize=12, usetex=True)
    
    plt.savefig('output.png')
    plt.show() 
    print("done")


if __name__ == '__main__':
    main()