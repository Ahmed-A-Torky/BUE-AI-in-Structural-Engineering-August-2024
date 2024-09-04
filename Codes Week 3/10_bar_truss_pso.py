# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 01:11:55 2020

@author: Ahmed_A_Torky
# https://pyswarms.readthedocs.io/en/latest/
# https://openseespydoc.readthedocs.io/en/latest
# https://github.com/cslotboom
# https://github.com/DEAP/deap
# 
Must install:
    pip install openseespy
    pip install pyswarms

Info:
# https://cdn.intechopen.com/pdfs/510/InTech-Particle_swarm_optimization_in_structural_design.pdf
The material used in this structure has a modulus of elasticity of and a mass
density of γ = 2.768T m3 . The design constraints are: maximum allowable stress in any member of
the truss σallowable = ±0.172MPa ; maximum deflection of any node (in both vertical and horizontal
directions) δallowable = 50.8mm . The upper and lower limits of cross-section areas of each truss
member 0.645mm2 ≤ Ai ≤ 225.8mm2 . The sample points by which the sample grid is generated are
1.0 and 30.0. 

"""
import openseespy.opensees as ops
import openseespy.opensees as op
import numpy as np
import matplotlib.pyplot as plt
import opsvis as opsv
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
P3 = -45.359*ton # 444.82 kN # 444822.1615 N
P5 = -45.359*ton # 444.82 kN # 444822.1615 N
# Weight
gamma = 2.76799*ton/(m**3)
# Lengths
L = []
for i in range(6):
    L.append(9.144*m)
for i in range(4):
    L.append(np.sqrt((9.144*m)**2 + (9.144*m)**2))
# =============================================================================
# Subroutine for 10-bar truss calculations
# =============================================================================
# Function of 10-bar truss
def tenbar_truss(x):
    # u = []
    Weights = []
    for ix in range(x.shape[0]):
        A=x[ix].tolist()
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
        ok = op.analyze(1)
        
            
        # Visualize
        # opsv.plot_model()
        sfac = 16.0e0
        # fig_wi_he = 22., 14.
        fig_wi_he = 30., 20.
        # # - 1
        nep = 9
        opsv.plot_defo(sfac, nep, az_el=(-68., 39.),
                        fig_wi_he=fig_wi_he, endDispFlag=0)
        plt.ylim(-2000, 10000)
        plt.title('3d 3-element cantilever truss')
        # # - 2
        # opsv.plot_defo(sfac, 19, az_el=(-125., 20.), fig_wi_he=fig_wi_he)
        # plt.title('3d 3-element deflection cantilever truss')
        # # - 3
        # sfacN = 0.002
        # opsv.section_force_diagram_2d('N', sfacN)
        # plt.title('Axial force distribution')
        plt.show()
        
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

        
        op.wipe()
     
        # ops.wipe()
    
        # ops.model('basic', '-ndm', 3, '-ndf', 6)
        # b = 0.1
        # h = 0.4
        # Iz, Iy, J = 0.0010667, 0.0002667, 0.01172
        # E1 = 25.0e6
        # G = 9615384.6
        # ops.node(1, 0.0, 0., 0.0)
        # ops.node(2, 0.0, 0., 9.144)
        # ops.node(3, 9.144, 0., 0.0)
        # ops.node(4, 9.144, 0., 9.144)
        # ops.node(5, 9.144*2, 0., 0.0)
        # ops.node(6, 9.144*2, 0., 9.144)
        # ops.fix(1, 1, 1, 1, 1, 1, 1)
        # ops.fix(2, 1, 1, 1, 1, 1, 1)
        # lmass = 200.
        # ops.mass(3, lmass, lmass, lmass, 0.001, 0.001, 0.001)
        # ops.mass(4, lmass, lmass, lmass, 0.001, 0.001, 0.001)
        # ops.mass(5, lmass, lmass, lmass, 0.001, 0.001, 0.001)
        # ops.mass(6, lmass, lmass, lmass, 0.001, 0.001, 0.001)
        # gTTagz = 1
        # gTTagx = 2
        # gTTagy = 3
        # coordTransf = 'Linear'
        # ops.geomTransf(coordTransf, gTTagz, 0., 1., 0.)
        # ops.geomTransf(coordTransf, gTTagx, 0., 1., 0.)
        # ops.geomTransf(coordTransf, gTTagy, 0., 1., 0.)
            
        # ops.element('elasticBeamColumn', 1, 1, 3, A[0], E1, G, J, Iy, Iz, gTTagz)
        # ops.element('elasticBeamColumn', 2, 3, 5, A[1], E1, G, J, Iy, Iz, gTTagz)
        # ops.element('elasticBeamColumn', 3, 5, 6, A[2], E1, G, J, Iy, Iz, gTTagz)
        # ops.element('elasticBeamColumn', 4, 6, 4, A[3], E1, G, J, Iy, Iz, gTTagz)
        # ops.element('elasticBeamColumn', 5, 4, 2, A[4], E1, G, J, Iy, Iz, gTTagz)
        # ops.element('elasticBeamColumn', 6, 3, 4, A[5], E1, G, J, Iy, Iz, gTTagz)
        # ops.element('elasticBeamColumn', 7, 1, 4, A[6], E1, G, J, Iy, Iz, gTTagz)
        # ops.element('elasticBeamColumn', 8, 2, 3, A[7], E1, G, J, Iy, Iz, gTTagz)
        # ops.element('elasticBeamColumn', 9, 3, 6, A[8], E1, G, J, Iy, Iz, gTTagz)
        # ops.element('elasticBeamColumn', 10, 5, 4, A[9], E1, G, J, Iy, Iz, gTTagz)
        
        # Ew = {}
        # Px = -4.e1
        # Py = -2.5e1
        # Pz = -3.e1
        # ops.timeSeries('Constant', 1)
        # ops.pattern('Plain', 1, 1)
        # ops.load(4, Px, Py, Pz, 0., 0., 0.)
        # ops.constraints('Transformation')
        # ops.numberer('RCM')
        # ops.system('BandGeneral')
        # ops.test('NormDispIncr', 1.0e-6, 6, 2)
        # ops.algorithm('Linear')
        # ops.integrator('LoadControl', 1)
        # ops.analysis('Static')
        # # ops.analyze(1)
        # # just for demonstration,
        # # the section data below does not match the data in OpenSees model above
        # # For now it can be source of inconsistency because OpenSees has
        # # not got functions to return section dimensions.
        # # A workaround is to have own Python helper functions to reuse data
        # # specified once
        # factor = 200
        # ele_shapes = {1: ['rect', [np.sqrt(A[0])/factor, np.sqrt(A[0])/factor]],
        #               2: ['rect', [np.sqrt(A[1])/factor, np.sqrt(A[1])/factor]],
        #               3: ['rect', [np.sqrt(A[2])/factor, np.sqrt(A[2])/factor]],
        #               4: ['rect', [np.sqrt(A[3])/factor, np.sqrt(A[3])/factor]],
        #               5: ['rect', [np.sqrt(A[4])/factor, np.sqrt(A[4])/factor]],
        #               6: ['rect', [np.sqrt(A[5])/factor, np.sqrt(A[5])/factor]],
        #               7: ['rect', [np.sqrt(A[6])/factor, np.sqrt(A[6])/factor]],
        #               8: ['rect', [np.sqrt(A[7])/factor, np.sqrt(A[7])/factor]],
        #               9: ['rect', [np.sqrt(A[8])/factor, np.sqrt(A[8])/factor]],
        #              10: ['rect', [np.sqrt(A[9])/factor, np.sqrt(A[9])/factor]]}
        
        # fig_wi_he = 50., 60.
        # opsv.plot_extruded_shapes_3d(ele_shapes, az_el=(-95., 5.), fig_wi_he=fig_wi_he)
        plt.show()
    
    # u = np.array(u)
    Weights = np.array(Weights)
    # print(Weights)
    return Weights/lb
# =============================================================================
#                                   PSO
# =============================================================================
from pyswarms.single.global_best import GlobalBestPSO
from pyswarms.utils.plotters import plot_cost_history
# instatiate the optimizer
x_max = A1 * np.ones(10)
x_min = A2 * np.ones(10)
bounds = (x_min, x_max)

Particles = 5
Iteration  = 500


# options = {'c1': 1.0, 'c2': 0.5, 'w': 0.95}
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
optimizer = GlobalBestPSO(n_particles=Particles, dimensions=10, options=options, bounds=bounds)
# now run the optimization
cost, pos = optimizer.optimize(tenbar_truss, Iteration)
# Try to get Weight less than 5100 lb
print("Final Weight is:","{:.2f}".format(round(cost, 2)),"lb")
print("Best cross-sections are:",pos/inch**2,"inches**2")
# Obtain cost history from optimizer instance
cost_history = optimizer.cost_history
# Plot Cost Decrease
plot_cost_history(cost_history)
plt.show()


"""
Our first example considers a well-known problem corresponding to a 10-bar truss nonconvex optimization shown on Fig. 6 with nodal coordinates and loading as shown in Table
1 and 2 (Sunar & Belegundu, 1991). In this problem the cross-sectional area for each of the 10
members in the structure are being optimized towards the minimization of total weight. The
cross-sectional area varies between 0.1 to 35.0 in2. Constraints are specified in terms of stress
and displacement of the truss members. The allowable stress for each member is 25,000 psi
for both tension and compression, and the allowable displacement on the nodes is ±2 in, in
the x and y directions. The density of the material is 0.1 lb/in3, Young’s modulus is E = 104
ksi and vertical downward loads of 100 kips are applied at nodes 2 and 4. In total, the
problem has a variable dimensionality of 10 and constraint dimensionality of 32 (10 tension
constraints, 10 compression constraints, and 12 displacement constraints). 
"""
