# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 18:56:12 2020
Must install:
    pip install geneticalgorithm
    pip install openseespy
@author: Ahmed_A_Torky
"""
import openseespy.opensees as op
import openseespy.opensees as ops
import numpy as np
import matplotlib.pyplot as plt
import opsvis as opsv

kji = 0

# =============================================================================
# Units
# =============================================================================
mm = 1.
N = 1.
Pa = 1.

inch = 25.4*mm
m = 1000.*mm
kN = 1000.*N
MPa = 1. # (10.**6)*Pa
GPa = (10.**3)*MPa
ton = 9.80665*kN
lb = 4.4482*N
# =============================================================================
# Input Variables
# =============================================================================
# Node Coordinates
x1 = 0.
y1 = 0.
x2 = 0.
y2 = 9.144*m
x3 = 9.144*m
y3 = 0.
x4 = 9.144*m
y4 = 9.144*m
x5 = 9.144*m * 2
y5 = 0.
x6 = 9.144*m * 2
y6 = 9.144*m
# Section Area (0.645mm**2 ≤ Ai ≤ 225.8mm**2)
A1 = 35.0*inch**2 # 225.8 # mm**2
A2 = 0.10*inch**2 # 0.645 # mm**2
# Modulus of elasticity
E = 68.94757*GPa # 68947.5908 MPa
# Loads
P3 = -45.359*ton 
P5 = -45.359*ton 
# Weight
gamma = 2.76799*ton/(m**3)
# Lengths
L = []
for i in range(6):
    L.append(9.144*m)
for i in range(4):
    L.append(np.sqrt((9.144*m)**2 + (9.144*m)**2))
    
kji = 0

# =============================================================================
# Subroutine for 10-bar truss calculations
# =============================================================================
# Function of 10-bar truss
def tenbar_truss(x):
    x = np.array(x)
    # print(x)
    # u = []
    Weights = []
    A=x.tolist()
    # print(A)
    # =============================================================================
    # OpenSees Analysis
    # =============================================================================
    # remove existing model
    op.wipe()
    # set modelbuilder
    op.model('basic', '-ndm', 2, '-ndf', 3)
    # define materials
    op.uniaxialMaterial("Elastic", 1, E)
    # create nodes
    op.node(1, x1, y1)
    op.node(2, x2, y2)
    op.node(3, x3, y3)
    op.node(4, x4, y4)
    op.node(5, x5, y5)
    op.node(6, x6, y6)
    # set boundary condition
    op.fix(1, 1, 1, 1)
    op.fix(2, 1, 1, 1)
    op.fix(3, 0, 0, 1)
    op.fix(4, 0, 0, 1)
    op.fix(5, 0, 0, 1)
    op.fix(6, 0, 0, 1)
    # define elements
    # op.element('Truss', eleTag, *eleNodes, A, matTag[, '-rho', rho][, '-cMass', cFlag][, '-doRayleigh', rFlag])
    op.element("Truss", 1, 1, 3, A[0], 1)
    op.element("Truss", 2, 3, 5, A[1], 1)
    op.element("Truss", 3, 5, 6, A[2], 1)
    op.element("Truss", 4, 6, 4, A[3], 1)
    op.element("Truss", 5, 4, 2, A[4], 1)
    op.element("Truss", 6, 3, 4, A[5], 1)
    op.element("Truss", 7, 1, 4, A[6], 1)
    op.element("Truss", 8, 2, 3, A[7], 1)
    op.element("Truss", 9, 3, 6, A[8], 1)
    op.element("Truss", 10, 5, 4, A[9], 1)
    # create TimeSeries
    op.timeSeries("Linear", 1)
    # create a plain load pattern
    op.pattern("Plain", 1, 1)
    # Create the nodal load - command: load nodeID xForce yForce
    # op.load(4, Px, Py, 0.)
    op.load(3, 0., P3, 0.)
    op.load(5, 0., P5, 0.)
    # No need to Record Results (writing takes time)
    # create SOE
    op.system("BandSPD")
    # create DOF number
    op.numberer("RCM")
    # create constraint handler
    op.constraints("Plain")
    # create integrator
    op.integrator("LoadControl", 1.0)
    # create algorithm
    op.algorithm("Newton")
    # create analysis object
    op.analysis("Static")
    # perform the analysis
    op.initialize() 
    op.analyze(1)
    
    # # Visualize
    # # opsv.plot_model()
    # sfac = 10.0e0
    # # fig_wi_he = 22., 14.
    # fig_wi_he = 30., 20.
    # # - 1
    # nep = 9
    # opsv.plot_defo(sfac, nep, az_el=(-68., 39.),
    #                fig_wi_he=fig_wi_he, endDispFlag=0)
    # plt.title('3d 3-element cantilever truss')
    # # # - 2
    # # opsv.plot_defo(sfac, 19, az_el=(-125., 20.), fig_wi_he=fig_wi_he)
    # # plt.title('3d 3-element deflection cantilever truss')
    # sfacN = 0.002

    # opsv.section_force_diagram_2d('N', sfacN)
    # plt.title('Axial force distribution')
    # plt.show()
    
    # Check the vl & hl results of displacement
    ux = []
    uy = []
    for i in range(6):
        ux.append(op.nodeDisp(i+1,1))
        uy.append(op.nodeDisp(i+1,2))
    # Must reset
    op.wipe()
    TotalWegiht = 0.0
    for ii in range(10):
        TotalWegiht += gamma*L[ii]*A[ii]
    if any(x >= 50.8 for x in np.abs(ux)):
        # print("ux exceeded 50.8mm")
        Weights.append(100000)
    elif any(y >= 50.8 for y in np.abs(uy)):
        # print("uy exceeded 50.8mm")
        Weights.append(100000)
    else:
        Weights.append(TotalWegiht)
    # u = np.array(u)
    Weights = np.array(Weights)
    # print(Weights/lb)

    op.wipe()
 
    ops.wipe()

    ops.model('basic', '-ndm', 3, '-ndf', 6)
    b = 0.1
    h = 0.4
    Iz, Iy, J = 0.0010667, 0.0002667, 0.01172
    E1 = 25.0e6
    G = 9615384.6
    ops.node(1, 0.0, 0., 0.0)
    ops.node(2, 0.0, 0., 9.144)
    ops.node(3, 9.144, 0., 0.0)
    ops.node(4, 9.144, 0., 9.144)
    ops.node(5, 9.144*2, 0., 0.0)
    ops.node(6, 9.144*2, 0., 9.144)
    ops.fix(1, 1, 1, 1, 1, 1, 1)
    ops.fix(2, 1, 1, 1, 1, 1, 1)
    lmass = 200.
    ops.mass(3, lmass, lmass, lmass, 0.001, 0.001, 0.001)
    ops.mass(4, lmass, lmass, lmass, 0.001, 0.001, 0.001)
    ops.mass(5, lmass, lmass, lmass, 0.001, 0.001, 0.001)
    ops.mass(6, lmass, lmass, lmass, 0.001, 0.001, 0.001)
    gTTagz = 1
    gTTagx = 2
    gTTagy = 3
    coordTransf = 'Linear'
    ops.geomTransf(coordTransf, gTTagz, 0., 1., 0.)
    ops.geomTransf(coordTransf, gTTagx, 0., 1., 0.)
    ops.geomTransf(coordTransf, gTTagy, 0., 1., 0.)
    ops.element('elasticBeamColumn', 1, 1, 3, A[0], E1, G, J, Iy, Iz, gTTagz)
    ops.element('elasticBeamColumn', 2, 3, 5, A[1], E1, G, J, Iy, Iz, gTTagz)
    ops.element('elasticBeamColumn', 3, 5, 6, A[2], E1, G, J, Iy, Iz, gTTagz)
    ops.element('elasticBeamColumn', 4, 6, 4, A[3], E1, G, J, Iy, Iz, gTTagz)
    ops.element('elasticBeamColumn', 5, 4, 2, A[4], E1, G, J, Iy, Iz, gTTagz)
    ops.element('elasticBeamColumn', 6, 3, 4, A[5], E1, G, J, Iy, Iz, gTTagz)
    ops.element('elasticBeamColumn', 7, 1, 4, A[6], E1, G, J, Iy, Iz, gTTagz)
    ops.element('elasticBeamColumn', 8, 2, 3, A[7], E1, G, J, Iy, Iz, gTTagz)
    ops.element('elasticBeamColumn', 9, 3, 6, A[8], E1, G, J, Iy, Iz, gTTagz)
    ops.element('elasticBeamColumn', 10, 5, 4, A[9], E1, G, J, Iy, Iz, gTTagz)
    
    Ew = {}
    Px = -4.e1
    Py = -2.5e1
    Pz = -3.e1
    ops.timeSeries('Constant', 1)
    ops.pattern('Plain', 1, 1)
    ops.load(4, Px, Py, Pz, 0., 0., 0.)
    ops.constraints('Transformation')
    ops.numberer('RCM')
    ops.system('BandGeneral')
    ops.test('NormDispIncr', 1.0e-6, 6, 2)
    ops.algorithm('Linear')
    ops.integrator('LoadControl', 1)
    ops.analysis('Static')
    # ops.analyze(1)
    # opsv.plot_model()
    # sfac = 2.0e0
    # fig_wi_he = 30., 20.
    # # - 1
    # nep = 9
    # opsv.plot_defo(sfac, nep, az_el=(-68., 39.),
    #                fig_wi_he=fig_wi_he, endDispFlag=0)
    # plt.title('3d 3-element cantilever beam')
    # # - 2
    # opsv.plot_defo(sfac, 19, az_el=(6., 30.), fig_wi_he=fig_wi_he)
    # plt.title('3d 3-element cantilever beam')
    # # - 3
    # nfreq = 6
    # eigValues = ops.eigen(nfreq)
    # modeNo = 6
    # sfac = 2.0e1
    # opsv.plot_mode_shape(modeNo, sfac, 19, az_el=(106., 46.),
    #                      fig_wi_he=fig_wi_he)
    # plt.title(f'Mode {modeNo}')
    # sfacN = 1.e-2
    # sfacVy = 5.e-2
    # sfacVz = 1.e-2
    # sfacMy = 1.e-2
    # sfacMz = 1.e-2
    # sfacT = 1.e-2
    # # plt.figure()
    # opsv.section_force_diagram_3d('N', sfacN)
    # plt.title('Axial force N')
    # opsv.section_force_diagram_3d('Vy', sfacVy)
    # plt.title('Transverse force Vy')
    # opsv.section_force_diagram_3d('Vz', sfacVz)
    # plt.title('Transverse force Vz')
    # opsv.section_force_diagram_3d('My', sfacMy)
    # plt.title('Bending moments My')
    # opsv.section_force_diagram_3d('Mz', sfacMz)
    # plt.title('Bending moments Mz')
    # opsv.section_force_diagram_3d('T', sfacT)
    # plt.title('Torsional moment T')
    
    # just for demonstration,
    # the section data below does not match the data in OpenSees model above
    # For now it can be source of inconsistency because OpenSees has
    # not got functions to return section dimensions.
    # A workaround is to have own Python helper functions to reuse data
    # specified once
    factor = 200
    ele_shapes = {1: ['rect', [np.sqrt(A[0])/factor, np.sqrt(A[0])/factor]],
                  2: ['rect', [np.sqrt(A[1])/factor, np.sqrt(A[1])/factor]],
                  3: ['rect', [np.sqrt(A[2])/factor, np.sqrt(A[2])/factor]],
                  4: ['rect', [np.sqrt(A[3])/factor, np.sqrt(A[3])/factor]],
                  5: ['rect', [np.sqrt(A[4])/factor, np.sqrt(A[4])/factor]],
                  6: ['rect', [np.sqrt(A[5])/factor, np.sqrt(A[5])/factor]],
                  7: ['rect', [np.sqrt(A[6])/factor, np.sqrt(A[6])/factor]],
                  8: ['rect', [np.sqrt(A[7])/factor, np.sqrt(A[7])/factor]],
                  9: ['rect', [np.sqrt(A[8])/factor, np.sqrt(A[8])/factor]],
                 10: ['rect', [np.sqrt(A[9])/factor, np.sqrt(A[9])/factor]]}
    
    fig_wi_he = 50., 60.
    opsv.plot_extruded_shapes_3d(ele_shapes, az_el=(-95., 5.), fig_wi_he=fig_wi_he)
    # plt.savefig('img_video/1.png')
    # plt.savefig('foo.png')

    # kji = kji + 1
    plt.show()

    return Weights/lb
# =============================================================================
#                                   GA
# =============================================================================
# Genetic Algorithm Part
from geneticalgorithm import geneticalgorithm as ga
# Set the section bounds
x_max = A1 * np.ones(10)
x_min = A2 * np.ones(10)
varbound=np.array([[A2,A1]]*10)

Chromosome = 10
Iteration  = 50

# instatiate the optimizer
algorithm_param = {'max_num_iteration': Chromosome,\
                   'population_size':Iteration,\
                   'mutation_probability':0.1,\
                   'elit_ratio': 0.01,\
                   'crossover_probability': 0.2,\
                   'parents_portion': 0.3,\
                   'crossover_type':'uniform',\
                   'max_iteration_without_improv':10000}
# Define the model
model=ga(function=tenbar_truss,\
            dimension=10,\
            variable_type='real',\
            variable_boundaries=varbound,\
            algorithm_parameters=algorithm_param)
# Run the GA model
model.run()
# Try to get Weight less than 5100 lb (competitive)
print("Model parameters are:",algorithm_param)
print("Final Weight is:","{:.2f}".format(round(model.best_function, 2)),"lb")
print("Best cross-sections are:",model.best_variable/inch**2,"inches")