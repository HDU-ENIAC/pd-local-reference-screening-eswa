"""Signal models used in the paper. Frequencies are cycles/sample; time is samples."""
import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt


def prepare_background(raw):
    raw = np.asarray(raw, dtype=np.float32).copy()
    if raw.shape != (800000,) or not np.isfinite(raw).all():
        raise ValueError("Expected one finite 800000-sample recording")
    raw -= raw.mean()
    sos = butter(5, 20000, fs=40000000, btype="highpass", output="sos")
    raw = sosfiltfilt(sos, raw).astype(np.float32)
    raw /= max(float(np.median(abs(raw-np.median(raw))))/.67449, 1e-4)
    return raw


def mode(t, f, tau):
    return (t >= 0)*np.exp(-np.maximum(t, 0)/tau)*np.sin(2*np.pi*f*np.maximum(t, 0))


def normal_response(g, rng, changed):
    t = np.arange(512, dtype=float)-96-rng.uniform(-2, 2)
    freq, tau = rng.uniform(.985, 1.015), rng.uniform(.95, 1.05)
    v = mode(t, g['f']*freq, g['tau']*tau)
    v += (.35+rng.uniform(-.05, .05))*mode(t, min(.23, 1.5*g['f'])*freq, .65*g['tau']*tau)
    if changed:
        v += .2*mode(t, .7*g['f']*freq, 1.4*g['tau']*tau)
        v *= 1-np.exp(-np.maximum(t, 0)/3.)
    return (g['amp']*rng.uniform(.9, 1.1)*v).astype(np.float32)


def pd_pulse(freq, tau, alternative=False):
    rate = 4
    t = np.arange(256*rate, dtype=float)/rate-32.
    u = np.maximum(t, 0.)
    phase = np.pi/3 if alternative else 0.
    h = np.exp(-u/tau)*np.sin(2*np.pi*freq*u+phase)*(t >= 0)
    if alternative:
        h *= 1-np.exp(-u/2.5)
        delay = 17*rate
        h[delay:] += -.4*h[:-delay].copy()
    h = resample_poly(h, 1, rate)
    h /= max(np.max(np.abs(h)), 1e-12)
    return h.astype(np.float32)


def main_conditions(raw, record):
    """Regenerate four main-evaluation contexts from a public background record.

    Record is the archived metadata row ID, not the measurement filename.
    Yields (condition, arrays). Only the main-evaluation generation is exposed
    here; the distinct historical source-training construction is not replaced.
    """
    raw = prepare_background(raw)
    grid = np.arange(1024, len(raw)-1024, 512)
    rng = np.random.default_rng(np.random.SeedSequence([710211, record]))
    chosen = []
    for k in rng.permutation(grid):
        if all(abs(k-j) >= 1024 for j in chosen):
            chosen.append(int(k))
        if len(chosen) == 40:
            break
    positions = np.array(chosen)
    eligible = grid[~((grid[:, None] < positions[None, :]+1024) &
                      (grid[:, None]+512 > positions[None, :]-512)).any(1)]
    rng = np.random.default_rng(np.random.SeedSequence([710213, record]))
    calpos = rng.choice(eligible, 512, replace=False).reshape(4, 128)
    noise = np.stack([raw[k:k+512] for k in positions])
    for changed in (False, True):
        for alternative in (False, True):
            rng = np.random.default_rng(np.random.SeedSequence([710212, record]))
            g = dict(f=float(rng.uniform(.025, .16)), tau=float(rng.uniform(12, 48)))
            t = np.arange(512)-96
            nominal = mode(t, g['f'], g['tau'])+.35*mode(t, min(.23, 1.5*g['f']), .65*g['tau'])
            if changed:
                nominal += .2*mode(t, .7*g['f'], 1.4*g['tau'])
                nominal *= 1-np.exp(-np.maximum(t, 0)/3.)
            peak = int(np.argmax(abs(nominal)))
            g['amp'] = float(np.sqrt(100*max(float(np.mean(noise[:8, peak-5:peak+35]**2)), 1e-8)/
                                     np.mean(nominal[peak-5:peak+35]**2)))
            refs = np.stack([noise[k]+normal_response(g, rng, changed) for k in range(8)])
            queries, components = [], []
            for k in range(32):
                q = noise[8+k]+normal_response(g, rng, changed)
                f = float(rng.uniform(.025, .22))
                tau = float(np.exp(rng.uniform(np.log(4), np.log(32))))
                h = pd_pulse(f, tau, alternative)
                start = 64+int(rng.integers(-48, 49))
                peak = int(np.argmax(abs(h)))
                snr = 0 if k % 2 == 0 else 5
                power = max(float(np.mean(noise[8+k, start+peak-5:start+peak+35].astype(float)**2)), 1e-8)
                amp = float(np.sqrt(power*10**(snr/10)/np.mean(h[peak-5:peak+35].astype(float)**2)))
                amp *= int(rng.choice([-1, 1]))
                event = np.zeros(512, np.float32)
                event[start:start+256] = amp*h
                queries.extend([q, q+event])
                components.extend([np.zeros(512, np.float32), event])
            calibration = []
            for bank in range(4):
                crng = np.random.default_rng(np.random.SeedSequence([710214, record, bank]))
                calibration.append(np.stack([raw[k:k+512]+normal_response(g, crng, changed)
                                             for k in calpos[bank]]))
            condition = ('nuisance_shift' if changed else 'nominal')+('_event_shift' if alternative else '')
            yield condition, dict(refs=refs.astype(np.float32), x=np.stack(queries),
                component=np.stack(components), noise=noise, positions=positions,
                calibration=np.stack(calibration), calibration_positions=calpos,
                y=np.tile([0, 1], 32), pair=np.repeat(np.arange(32), 2),
                snr=np.repeat([0 if k % 2 == 0 else 5 for k in range(32)], 2))
