import numpy as np

def euler_step(f, t, y, dt):        #f(t,y) is the derivative (derivs) function dy/dt = [vx, vy, ax, ay]
    return y + dt * f(t, y)         # = y_{n+1}

def rk4_step(f, t, y, dt):                      #improves accuracy by sampling the slope multiple times
    k1 = f(t, y)                                #slope at start
    k2 = f(t + 0.5*dt, y + 0.5*dt*k1)           #slope at the midpoint
    k3 = f(t + 0.5*dt, y + 0.5*dt*k2)           #corrected midpoint slope
    k4 = f(t + dt, y + dt*k3)                   #slope at the end
    return y + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)       #weighted average of slopes

def leapfrog_step(f, t, y, dt):             #symplectic, conserves phase-space structure
    r = y[:2]           #extracts position [x,y]
    v = y[2:]           #extracts velocity [vx,vy]

    a = f(t, y)[2:]     #[ax,ay] from [vx, vy, ax, ay]
    v_half = v + 0.5 * dt * a           #improves velocity by half step
    r_new = r + dt * v_half             #new position with half step velocity

    y_half = np.hstack((r_new, v_half))     #temp state
    a_new = f(t + dt, y_half)[2:]           #accelenration at new position
    v_new = v_half + 0.5 * dt * a_new       #complete velocity update

    return np.hstack((r_new, v_new))        #updated state vector

def integrate(f, y0, t0, tf, dt, method="rk4"):     #y0=[x, y, vx, vy] ; t0 initial time; tf final time
    step = {
        "euler": euler_step,
        "rk4": rk4_step,
        "leapfrog": leapfrog_step
    }[method]

    t = t0
    y = y0.copy()
    T, Y = [t], [y]     #list of time values and state vectors

    while t < tf:
        y = step(f, t, y, dt)       # y -> y_next from the chosen step form above
        t += dt
        T.append(t)         #adds new time ~(N) and state ~(N,4) N- number of timesteps
        Y.append(y)

    return np.array(T), np.array(Y)
