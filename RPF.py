import deepxde as dde
import numpy as np
from deepxde.backend import tf

dde.config.set_default_float("float64")
tf.random.set_random_seed(1234)


clight = 2.99792e10 # speed of lightin cm/s
CLASSICALER = 2.8179e-13 # classical electron radius in units of cm
mecSQ = 511e3 # electron rest mass in units eV
IA = 0.017045e6 # Alfven current in Amperes

epochsADAM = 15000
epochsBFGS = 100000
NumBFGS = 100
lr = 5.e-4
pts = 1000000
interiorpts = [pts,round(pts/4)]

T0 = 0 # Plasma temperature in eV

EFMin = -10
EFMax = -1

ZeffMin = 1
ZeffMax = 10

alphaMin = 0
alphaMax = 0.2

EnergyMaxeV = 5e6
EnergyMineV = 1e4
gMax = 1 + EnergyMaxeV / mecSQ
gMin = 1 + EnergyMineV / mecSQ

pMax = np.sqrt(gMax**2-1)
pMin = np.sqrt(gMin**2-1)
xiMax = 1
xiMin = -1


def save_solution(geom, model, filename):
    x = geom.uniform_points(40**3)
    y_pred = model.predict(x)
    print("Saving u and p ...\n")
    #np.savetxt(filename + "_fine.dat", np.hstack((x, y_pred, alpha(y_pred[:, -1:]))))
    np.savetxt(filename + "_fine.dat", np.hstack((x, y_pred)))

    #x = geom.uniform_points(4096)
    x = geom.uniform_points(20**3)
    y_pred = model.predict(x)
    print("Saving u and p ...\n")
    np.savetxt(filename + "_coarse.dat", np.hstack((x, y_pred)))


def pde(inputs, outputs):
    dy_p = dde.grad.jacobian(outputs, inputs, i=0, j=0)
    dy_pp = dde.grad.hessian(outputs, inputs, i=0, j=0)
    dy_xi = dde.grad.jacobian(outputs, inputs, i=0, j=1)
    dy_xixi = dde.grad.hessian(outputs, inputs, i=1, j=1)

    p, xi, EFNorm, ZeffNorm, alphaNorm = inputs[:, 0:1], inputs[:, 1:2], inputs[:, 2:3], inputs[:, 3:4], inputs[:, 4:5]

    Ephi = EFMin + ( EFMax - EFMin ) * EFNorm
    Zeff = ZeffMin + ( ZeffMax - ZeffMin ) * ZeffNorm
    alpha = alphaMin + ( alphaMax - alphaMin ) * alphaNorm
    
    Te = T0

    gamma = tf.sqrt(1+p*p)
    CA = (Te/mecSQ) * gamma*gamma*gamma/p/p/p
    dCAdp = -3 * (Te/mecSQ) * gamma/p/p/p/p

    ElecticFieldTerms = -Ephi * ( xi*dy_p + ((1-xi**2)/p)*dy_xi )
    CollisionalTerms = (gamma*gamma/p**2)*dy_p - ((Zeff+1)/2)*(gamma/p**3)*( (1-xi**2)*dy_xixi - 2*xi*dy_xi ) - CA*dy_pp - ( 2*CA/p + dCAdp )*dy_p
    #CollisionalTerms = (gamma*gamma/p**2)*dy_p - ((Zeff+1)/2)*(gamma/p**3)*( (1-xi**2)*dy_xixi - 2*xi*dy_xi )
    RadiationTerms = alpha * ( gamma*p*(1-xi**2)*dy_p - xi*(1-xi**2)/gamma*dy_xi )
    #RadiationTerms = 0

    #loss = (1/(5.e-2+outputs[:, 0:1])) * (p**2/(1+p**2)) * ( ElecticFieldTerms + CollisionalTerms + RadiationTerms )
    loss = (p**2/(1+p**2)) * ( ElecticFieldTerms + CollisionalTerms + RadiationTerms )

    return loss


def output_transform(inputs, outputs):
    p, xi = inputs[:, 0:1], inputs[:, 1:2]
    EFNorm = inputs[:, 2:3]
    #alphaNorm = inputs[:, 4:5]

    Ephi = EFMin + ( EFMax - EFMin ) * EFNorm
    #alpha = alphaMin + ( alphaMax - alphaMin ) * alphaNorm
    
    gamma = tf.sqrt(p**2+1)
    #theta = vTeOverc**2 / 2

    #Up = xi*Ephi - (gamma*gamma/p**2) - alpha*gamma*p*(1-xi**2)
    #DxiUp = 0.25
    #Up = xi*Ephi - (gMax**2/pMax**2) - alpha*gMax*pMax*(1-xi**2)
    #DxiUp = 0.25*tf.sqrt(Ephi**2+1)
    #XiOut = 0.5 * ( 1 + tf.math.tanh(Up/DxiUp) )
    
    #Dp_t = Dp * (1 + 1e2*t*CA )
    
    DProb = 1
    #DShiftp = Dp*np.arctanh(1-2*DProb*np.sqrt(0.54931)) # Shifts solution so the step function is 0.5 at p=pRE

    #ProbpInit = 0.5 * ( 1 - tf.tanh((pRE-p+DShiftp)/Dp) )

    dEphi = 0.1
    Heaviside = 0.5 * ( 1 + tf.tanh((-1-Ephi)/dEphi) )
    
    #Dxi = 0.5
    #XiEnv = tf.exp(-(xi-xiMin)**2/Dxi**2)

    #Prob = 0.5 * ( 1 - tf.tanh((0.5*pMax-p)/Dp) ) + ((p-pMin)/(pMax-pMin)) * ((t-tMin)/(tMax-tMin)) * ((pMax-p*XiEnv)/(pMax-pMin)) * outputs[:, 0:1]
    #Prob = ProbpInit + ((p-pMin)/(pMax-pMin)) * ((t-tMin)/(tMax-tMin)) * ((pMax-p*XiOut)/(pMax-pMin)) * outputs[:, 0:1]
    Prob = Heaviside*((p-pMin)/(pMax-pMin)) * outputs[:, 0:1]

    #DProb = 0.25
    Prob = tf.tanh((Prob/DProb)*(Prob/DProb))

    return tf.concat((Prob), axis=1)


def main():
    geom = dde.geometry.Hypercube([pMin, xiMin, 0, 0, 0], [pMax, xiMax, 1, 1, 1])

    #geomInterval = dde.geometry.Interval(pMin, pMax)
    #Logo_points = geomInterval.log_uniform_points(interiorpts[0])

    #LowEnergy = dde.geometry.Rectangle([pMin, xiMin], [pMax, -0.5])
    #left_corner = dde.geometry.Rectangle([0, 0.8], [0.2, 1])
    #right_corner = dde.geometry.Rectangle([0.8, 0.8], [1, 1])

    #uniform_points = spatio_temporal_domain.random_points(interiorpts[0])
    #uniform_points_2 = spatio_temporal_domain_2.random_points(interiorpts[1])
    #LowEnergy_points = LowEnergy.random_points(interiorpts[1])
    #left_corner_points = left_corner.random_points(interiorpts[2])
    #right_corner_points = right_corner.random_points(interiorpts[3])

    #points = uniform_points
    #points = np.append(uniform_points, uniform_points_2, axis = 0)
    #points = np.append(uniform_points, LowEnergy_points, axis = 0)
    #points = np.append(points, left_corner_points, axis = 0)
    #points = np.append(points, right_corner_points, axis = 0)

    #net = dde.maps.PFNN([2] + [[16] * 4] * 2 + [4], "tanh", "Glorot normal")

    #print("points = " + str(points))
    #print("p points = " + str(points[:,0]))
    #print("Log p points = " + str(Logp_points))

    # Use log distribution of energy points
    #points[:,0] = Logp_points[:,0]

    #print("new log p points = " + str(points))

    net = dde.maps.FNN([5] + [64] * 6 + [1], "tanh", "Glorot normal")
    net.apply_output_transform(output_transform)

    """
    def func(x):
        Up = xi*Ephi - (gMax**2/pMax**2) - alpha*gMax*pMax*(1-xi**2)
        DxiUp = 0.25*tf.sqrt(Ephi**2+1)
        return 0.5 * ( 1 + tf.math.tanh(Up/DxiUp) )
    """
    
    def boundary(inputs, on_boundary):
        p = inputs[0]
        xi = inputs[1]
        EFNorm = inputs[2]
        alphaNorm = inputs[4]

        Ephi = EFMin + ( EFMax - EFMin ) * EFNorm
        alpha = alphaMin + ( alphaMax - alphaMin ) * alphaNorm
    
        Up = xi*Ephi - (gMax**2/pMax**2) - alpha*gMax*pMax*(1-xi**2)
        return on_boundary and np.isclose(p, pMax) and Up > 0
    
    
    bc_pMax = dde.DirichletBC(
        geom,
        lambda x: 1 + 0*x[:, 1:2],
        #psi_func,
        #lambda x, on_boundary: on_boundary, 
        #component=0
        boundary,
        #lambda x, on_boundary: on_boundary and np.isclose(x[0], pMax),
        component=0
        #lambda x, on_boundary: on_boundary and np.isclose(x[0], pMax) and x[1] < 0.25,
    )

    """
    bc_pMin = dde.DirichletBC(
        geom,
        lambda x: 0*x[:, 1:2],
        #psi_func,
        #lambda x, on_boundary: on_boundary, 
        #component=0
        #boundary,
        lambda x, on_boundary: on_boundary and np.isclose(x[0], pMin),
        component=0
        #lambda x, on_boundary: on_boundary and np.isclose(x[0], pMax) and x[1] < 0.25,
    )
    """    
    """
    bc_pMax = dde.NeumannBC(
        spatio_temporal_domain,
        lambda x: 0*x[:, 1:2],
        #psi_func,
        #lambda x, on_boundary: on_boundary, 
        #component=0
        lambda x, on_boundary: on_boundary and np.isclose(x[0], pMax),
        component=0
        #lambda x, on_boundary: on_boundary and np.isclose(x[0], pMax) and x[1] < 0.25,
    )
    """
    """
    ic = dde.IC(
        spatio_temporal_domain,
        psi_func,
        lambda _, on_initial: on_initial, component=0
)
    """
    #losses = [dde.OperatorBC(geom, dissipated_power, lambda x, _: not geom.on_boundary(x))]
    losses = [bc_pMax]
    #losses = [bc_pMax]
    #losses = []


    #pde_resampler = dde.callbacks.PDEPointResampler(period=100)
    data = dde.data.TimePDE(
        geom,
        pde,
        losses,
        num_domain=pts,
        num_boundary=10000,
        num_initial=0,
        num_test=pts,
        train_distribution='Hammersley',
        #anchors = points
    )

    """
    data = dde.data.PDE(
        geom,
        pde,
        losses,
        num_domain=2**10,
        num_boundary=2*8,
        num_test=2**13
    )
    """
    model = dde.Model(data, net)

    #loss_weights = [1] * 2
    #loss = ["MSE"] * 2
    #model.compile("adam", lr=lr, loss=loss, loss_weights=loss_weights)
    model.compile("adam", lr=lr)

    #model.compile("adam", lr=lr)
    #losshistory, train_state = model.train(epochs=epochsADAM, callbacks=[pde_resampler], model_save_path = './model/model.ckpt')
    losshistory, train_state = model.train(epochs=epochsADAM, model_save_path = './model/model.ckpt')

    for i in range(0,NumBFGS):
        #model.compile("L-BFGS-B", loss=loss, loss_weights=loss_weights)
        model.compile("L-BFGS-B")

        model.train_step.optimizer_kwargs = {'options': {'maxcor': 100,
                                                         'ftol': 1.0 * np.finfo(float).eps,
                                                         'gtol': 1.0 * np.finfo(float).eps,
                                                         'maxiter': epochsBFGS,
                                                         'maxfun':  epochsBFGS,
                                                         'maxls': 200}}
     
        #losshistory, train_state = model.train(callbacks=[pde_resampler], model_save_path = './model/model.ckpt')
        losshistory, train_state = model.train(model_save_path = './model/model.ckpt')




        xpp = geom.random_points(100000)
        #print("xpp = " + str(xpp))
        f = model.predict(xpp, operator=pde)
        err_eq_f = np.absolute(f)
        for i in range(0,10):
            x_id = np.argmax(err_eq_f)
            #print("Adding new points:", xpp[x_id], "\n")
            data.add_anchors(xpp[x_id])
            err_eq_f = np.delete(err_eq_f,x_id)

        """
        k=1 # increase to add more adaptivity
        c=1 # increase to make distribution of training points more uniform
        FracPts = 0.1 # add points in increments to avoid running out of memory
        NumPtsToAdd = round(FracPts*pts)

        xpp = spatio_temporal_domain.random_points(25*NumPtsToAdd)
        #print("xpp = " + str(xpp))
        ftmp = np.abs(model.predict(xpp, operator=pde)).astype(np.float64)
        f = ftmp
        #f0 = ftmp[0,:]
        #f1 = ftmp[1,:]
        #f2 = ftmp[2,:]
        #f = (f0+f1) / 2
        err_eq = np.power(f, k) / np.power(f, k).mean() + c
        err_eq_normalized = (err_eq / sum(err_eq))[:, 0]
        xpp_ids = np.random.choice(a=len(xpp), size=NumPtsToAdd, replace=False, p=err_eq_normalized)
        xpp_selected = xpp[xpp_ids]
        data.replace_with_anchors(xpp_selected)
        #err_eq_f = np.absolute(f)

        for i in range(0, round(1/FracPts)-1):
            xpp = spatio_temporal_domain.random_points(25*NumPtsToAdd)
            #print("xpp = " + str(xpp))
            ftmp = np.abs(model.predict(xpp, operator=pde)).astype(np.float64)
            f = ftmp
            #f0 = ftmp[0,:]
            #f1 = ftmp[1,:]
            #f2 = ftmp[2,:]
            #f = (f0+f1) / 2
            err_eq = np.power(f, k) / np.power(f, k).mean() + c
            err_eq_normalized = (err_eq / sum(err_eq))[:, 0]
            xpp_ids = np.random.choice(a=len(xpp), size=NumPtsToAdd, replace=False, p=err_eq_normalized)
            xpp_selected = xpp[xpp_ids]
            data.add_anchors(xpp_selected)
            #err_eq_f = np.absolute(f)

        """
        
        #save_solution(geom, model, "./data/solution0")

        dde.saveplot(losshistory, train_state, issave=True, isplot=True)


if __name__ == "__main__":
    main()
