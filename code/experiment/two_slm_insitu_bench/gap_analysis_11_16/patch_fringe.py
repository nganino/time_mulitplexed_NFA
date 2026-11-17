"""2026-11-16: add luna_fringe and spec_adjust (both off by default) to config.py / model.py. Backs up both first."""
import shutil, os
os.chdir(r'P:\Nicholas\time_multiplexed_NFA\code')
for f in ('config.py', 'model.py'):
    b = f.replace('.py', '_before_fringe_1116.py')
    if not os.path.exists(b):
        shutil.copy2(f, b)

c = open('config.py', encoding='utf-8').read()
old = "    tc.spec_P_mult    = 1.25\n"
assert c.count(old) == 1
c = c.replace(old, old + """    # 2026-11-16 (gap_analysis_11_16): bench non-idealities fitted to the 11-14 zero-shot, spectral propagator only.
    # luna_fringe w: LUNA pixel crosstalk, each displayed phase -> (1 - 4w) itself + w each of its 4 neighbours (fit 0.09).
    # spec_adjust: 'key_dy=..,key_dx=..' (PLUTO px), 'luna_dy/luna_dx' (LUNA px), 'det_dy/det_dx' (camera px),
    # 'cam_mag' (fraction, scales the detector grid and boxes); '' = trained geometry
    tc.luna_fringe    = 0.0
    tc.spec_adjust    = ''
""")
open('config.py', 'w', encoding='utf-8').write(c)

m = open('model.py', encoding='utf-8').read()
old = "            self.spec_geom = geom\n            self.spec = SpectralLabProp(config, geom)\n"
assert m.count(old) == 1
m = m.replace(old, """            adj = {k.strip(): float(v) for k, v in (kv.split('=') for kv in str(getattr(config, 'spec_adjust', '') or '').split(',') if kv.strip())}
            if adj:   # 2026-11-16: fitted bench geometry (config.spec_adjust)
                bad = set(adj) - {'key_dy', 'key_dx', 'luna_dy', 'luna_dx', 'det_dy', 'det_dx', 'cam_mag'}
                assert not bad, f'unknown spec_adjust keys {bad}'
                cm, cdx = 1 + adj.get('cam_mag', 0.0), config.camera_dx
                geom.update(key_y0=geom['key_y0'] + adj.get('key_dy', 0.0) * geom['p1'],
                            key_x0=geom['key_x0'] + adj.get('key_dx', 0.0) * geom['p1'],
                            luna_y0=geom['luna_y0'] + adj.get('luna_dy', 0.0) * geom['p2'],
                            luna_x0=geom['luna_x0'] + adj.get('luna_dx', 0.0) * geom['p2'],
                            det_cy=np.asarray(geom['det_cy']) * cm + adj.get('det_dy', 0.0) * cdx,
                            det_cx=np.asarray(geom['det_cx']) * cm + adj.get('det_dx', 0.0) * cdx,
                            det_size=geom['det_size'] * cm)
            self.luna_fringe = float(getattr(config, 'luna_fringe', 0.0) or 0.0)
""" + old)
old = "            d = torch.exp(1j * self._input_plane_px(a)) - 1                                       # [B, F, ny, nx]\n"
assert m.count(old) == 1
m = m.replace(old, "            d = torch.exp(1j * self._fringe(self._input_plane_px(a))) - 1                         # [B, F, ny, nx]\n")
old = "    def input_plane_field(self, a):\n"
assert m.count(old) == 1
m = m.replace(old, """    def _fringe(self, phi):
        # 2026-11-16: LUNA pixel crosstalk (config.luna_fringe w; spectral path only): each displayed phase becomes
        # (1 - 4w) itself + w each of its 4 neighbours (edge pixels replicate). Identity when w = 0.
        w = getattr(self, 'luna_fringe', 0.0)
        if not w:
            return phi
        k = phi.new_tensor([[0, w, 0], [w, 1 - 4 * w, w], [0, w, 0]])
        sh = phi.shape
        return F.conv2d(F.pad(phi.reshape(-1, 1, *sh[-2:]), (1, 1, 1, 1), mode='replicate'), k[None, None]).reshape(sh)

""" + old)
open('model.py', 'w', encoding='utf-8').write(m)
print('patched')
