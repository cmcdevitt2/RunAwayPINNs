##############################################################################
# Solves the time dependent adjoint of the relativistic Fokker-Planck equation
# for a range of electric field strengths, Z_eff, and synchrotron radiation
##############################################################################
import deepxde as dde
import numpy as np
from deepxde.backend import tf
from scipy.special import kn

dde.config.set_default_float("float64")
tf.random.set_random_seed(1234)

epochsADAM = 5000
epochsBFGS = 20000
NumBFGS = 20
lr = 5.e-4
pts = 200000

EnergyMaxeV = 5e6
EnergyMineV = 1e4

mecSQ = 511e3 # electron rest mass in units eV
gMax = 1 + EnergyMaxeV / mecSQ
gMin = 1 + EnergyMineV / mecSQ

pMax = np.sqrt(gMax**2-1)
pMin = np.sqrt(gMin**2-1)
xiMax = 1
xiMin = -1
tMax = 5 # final time in units of tau_c
tMin = 0

# sets momentum above which electrons are counted as REs
pRE = 0.25 * pMax

# Electric field convention is such that E_phi is negative
EFMax = -1 # normalized to E_c
EFMin = -3

ZeffMin = 1
ZeffMax = 2

alphaMin = 0
alphaMax = 0.1

Dp = 0.1*pMax # sets width of transition region in initial RPF


def pde(inputs, outputs):
    dy_p = dde.grad.jacobian(outputs, inputs, i=0, j=0) / (pMax-pMin)
    dy_xi = dde.grad.jacobian(outputs, inputs, i=0, j=1)
    dy_xixi = dde.grad.hessian(outputs, inputs, i=1, j=1)
    dy_t = dde.grad.jacobian(outputs, inputs, i=0, j=5) / (tMax-tMin)

    pNorm, xi = inputs[:, 0:1], inputs[:, 1:2]
    EFNorm = inputs[:, 2:3]
    ZeffNorm = inputs[:, 3:4]
    alphaNorm = inputs[:, 4:5]

    p = pMin + ( pMax - pMin ) * pNorm
    Ephi = EFMin + ( EFMax - EFMin ) * EFNorm
    Zeff = ZeffMin + ( ZeffMax - ZeffMin ) * ZeffNorm
    alpha = alphaMin + ( alphaMax - alphaMin ) * alphaNorm

    gamma = tf.sqrt(1+p*p)

    CF = gamma*gamma/p**2
    CB = 0.5*(Zeff+1)*gamma/p

    ElecticFieldTerms = -Ephi * ( xi*dy_p + ((1-xi**2)/p)*dy_xi )
    CollisionalTerms = CF*dy_p - (CB/p**2)*( (1-xi**2)*dy_xixi - 2*xi*dy_xi )
    RadiationTerms = alpha * ( gamma*p*(1-xi**2)*dy_p - xi*(1-xi**2)/gamma*dy_xi )

    DpMax = 0.05*pMax
    StepFunc = 1 - tf.exp(-(p-pMax)**2/DpMax**2)
    loss = StepFunc * (p**2/(1+p**2)) * ( dy_t + ElecticFieldTerms + CollisionalTerms + RadiationTerms )

    return loss


def output_transform(inputs, outputs):
    pNorm, xi = inputs[:, 0:1], inputs[:, 1:2]
    EFNorm = inputs[:, 2:3]
    ZeffNorm = inputs[:, 3:4]
    alphaNorm = inputs[:, 4:5]
    tNorm = inputs[:, 5:6]

    p = pMin + ( pMax - pMin ) * pNorm
    Ephi = EFMin + ( EFMax - EFMin ) * EFNorm
    alpha = alphaMin + ( alphaMax - alphaMin ) * alphaNorm
    t = tMin + ( tMax - tMin ) * tNorm
    
    DProbp = 0.15
    ProbpInit = tf.tanh((p-pRE)/Dp)/DProbp
    
    Probp = ProbpInit + (p-pMin)/(pMax-pMin) * tf.tanh(t) * outputs[:, 0:1]
    Probp = 0.5 * ( 1 + tf.tanh(Probp) )

    return tf.concat(Probp, axis=1)


def main():
    #                              pMin, xiMin, EFmin, Zeffmin, alphamin
    geom = dde.geometry.Hypercube([0,    xiMin, 0,     0,       0], [1, xiMax, 1, 1, 1])
    temporal_domain = dde.geometry.TimeDomain(0, 1)
    spatio_temporal_domain = dde.geometry.GeometryXTime(geom, temporal_domain)

    net = dde.maps.FNN([6] + [64] * 4 + [1], "tanh", "Glorot normal")
    net.apply_output_transform(output_transform)


    def boundary(inputs, on_boundary):
        pNorm = inputs[0]
        xi = inputs[1]
        EFNorm = inputs[2]
        alphaNorm = inputs[4]

        p = pMin + ( pMax - pMin ) * pNorm
        Ephi = EFMin + ( EFMax - EFMin ) * EFNorm
        alpha = alphaMin + ( alphaMax - alphaMin ) * alphaNorm

        Up = xi*Ephi - (gMax**2/pMax**2) - alpha*gMax*pMax*(1-xi**2)
        return on_boundary and np.isclose(p, pMax) and Up > 0
    
    
    bc_pMax = dde.DirichletBC(
        spatio_temporal_domain,
        lambda x: 1 + 0*x[:, 1:2],
        boundary,
        component=0
    )
    
    losses = [bc_pMax]

    data = dde.data.TimePDE(
        spatio_temporal_domain,
        pde,
        losses,
        num_domain=pts,
        num_boundary=round(pts/50),
        num_initial=0,
        num_test=pts,
        train_distribution='Hammersley',
    )

    model = dde.Model(data, net)

    loss_weights = [10] + [1]
    loss = ["MSE"] * 2
    model.compile("adam", lr=lr, loss=loss, loss_weights=loss_weights)

    losshistory, train_state = model.train(epochs=epochsADAM, model_save_path = './model/model.ckpt')

    resampler = dde.callbacks.PDEPointResampler(period=500)
    for i in range(0,NumBFGS):
        model.compile("L-BFGS-B", loss=loss, loss_weights=loss_weights)


        model.train_step.optimizer_kwargs = {'options': {'maxcor': 100,
                                                         'ftol': 1.0 * np.finfo(float).eps,
                                                         'gtol': 1.0 * np.finfo(float).eps,
                                                         'maxiter': epochsBFGS,
                                                         'maxfun':  epochsBFGS,
                                                         'maxls': 200,
                                                         'method':'BFGS'}}


        losshistory, train_state = model.train(model_save_path = './model/model.ckpt',callbacks=[resampler])


        # Residual based adaptive ressampling training points
        k=1 # increase to add more adaptivity
        c=1 # increase to make distribution of training points more uniform
        FracPts = 0.1 # add points in increments to avoid running out of memory
        NumPtsToAdd = round(FracPts*pts)

        xpp = spatio_temporal_domain.random_points(25*NumPtsToAdd)
        ftmp = np.abs(model.predict(xpp, operator=pde)).astype(np.float64)
        f = ftmp
        err_eq = np.power(f, k) / np.power(f, k).mean() + c
        err_eq_normalized = (err_eq / sum(err_eq))[:, 0]
        xpp_ids = np.random.choice(a=len(xpp), size=NumPtsToAdd, replace=False, p=err_eq_normalized)
        xpp_selected = xpp[xpp_ids]
        data.replace_with_anchors(xpp_selected)

        for i in range(0, round(1/FracPts)-1):
            xpp = spatio_temporal_domain.random_points(25*NumPtsToAdd)
            ftmp = np.abs(model.predict(xpp, operator=pde)).astype(np.float64)
            f = ftmp
            err_eq = np.power(f, k) / np.power(f, k).mean() + c
            err_eq_normalized = (err_eq / sum(err_eq))[:, 0]
            xpp_ids = np.random.choice(a=len(xpp), size=NumPtsToAdd, replace=False, p=err_eq_normalized)
            xpp_selected = xpp[xpp_ids]
            data.add_anchors(xpp_selected)


        dde.saveplot(losshistory, train_state, issave=True, isplot=True)


if __name__ == "__main__":
    main()
