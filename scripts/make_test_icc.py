#!/usr/bin/env python3
"""Build the output-profile test fixtures in tests/fixtures/icc/ with Little CMS.

The profiles describe a synthetic inkjet-like RGB printer, analytic so that
its device -> PCS and PCS -> device tables are exact inverses:

    lin   = device ** 2.2                         (per channel)
    lin'  = S * lin + (1 - S) * mean(lin)          (gamut reduced towards gray)
    XYZ   = K * D50 + (1 - K) * M_srgb_d50 . lin'  (media-relative PCS, black lifted)

with K the black of L* = 8 and a tinted paper white (wtpt) for absolute
colorimetry.  The intents differ: the perceptual PCS -> device table
compresses chroma by 0.85 (gamut compression), the saturation one expands
it by 1.1, and both scale lightness onto the printer's range (PCS black
L* = 0 is the device black), as perceptual tables do; the colorimetric
ones are the model itself, clipping below its black.  Each device -> PCS
table is the inverse of its PCS -> device one.

    printer-rgb-v4.icc  output class, v4.3, Lab PCS, lutAtoBType / lutBtoAType
    printer-rgb-v2.icc  the same as v2.4, lut16Type
    printer-gray-v4.icc output class, v4.3, gray TRC with the same black lift

Run from the repository root:  python3 scripts/make_test_icc.py
Needs liblcms2 (liblcms2-2).  The output is committed; the tests never run this.
"""

import ctypes
import ctypes.util
import math
import os
import sys

LCMS = ctypes.CDLL(ctypes.util.find_library("lcms2") or "liblcms2.so.2")

OUT = os.path.join("tests", "fixtures", "icc")

D50 = (0.9642, 1.0, 0.8249)
# D50-adapted sRGB colorants as columns (rXYZ, gXYZ, bXYZ).
M = (
    (0.4360747, 0.3850649, 0.1430804),
    (0.2225045, 0.7168786, 0.0606169),
    (0.0139322, 0.0971045, 0.7141733),
)
S = 0.8
L_BLACK = 8.0
PAPER = (0.93, 0.965, 0.76)
B2A_GRID = 25
A2B_GRID = 17


def lab_to_y(l):
    f = (l + 16.0) / 116.0
    return f**3 if f > 6.0 / 29.0 else 3 * (6.0 / 29.0) ** 2 * (f - 4.0 / 29.0)


K = lab_to_y(L_BLACK)


def inv3(m):
    a, b, c = m[0]
    d, e, f = m[1]
    g, h, i = m[2]
    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    return (
        ((e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det),
        ((f * g - d * i) / det, (a * i - c * g) / det, (c * d - a * f) / det),
        ((d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det),
    )


M_INV = inv3(M)


def mul(m, v):
    return tuple(sum(m[r][c] * v[c] for c in range(3)) for r in range(3))


def fwd(x):
    return x ** (1 / 3) if x > (6 / 29) ** 3 else x / (3 * (6 / 29) ** 2) + 4 / 29


def inv(t):
    return t**3 if t > 6 / 29 else 3 * (6 / 29) ** 2 * (t - 4 / 29)


def xyz_to_lab(xyz):
    fx, fy, fz = (fwd(xyz[i] / D50[i]) for i in range(3))
    return (116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz))


def lab_to_xyz(lab):
    l, a, b = lab
    fy = (l + 16) / 116
    return (D50[0] * inv(fy + a / 500), D50[1] * inv(fy), D50[2] * inv(fy - b / 200))


def lin_to_lab(lin, chroma, rescale):
    mean = sum(lin) / 3
    lin = [S * v + (1 - S) * mean for v in lin]
    xyz = mul(M, lin)
    xyz = tuple(K * D50[i] + (1 - K) * xyz[i] for i in range(3))
    l, a, b = xyz_to_lab(xyz)
    if rescale:
        l = (l - L_BLACK) * 100 / (100 - L_BLACK)
    return (l, a * chroma, b * chroma)


def device_to_lab(dev, chroma, rescale=False):
    return lin_to_lab([max(0.0, v) ** 2.2 for v in dev], chroma, rescale)


def lab_to_device(lab, chroma, rescale=False):
    return tuple(v ** (1 / 2.2) for v in lab_to_lin(lab, chroma, rescale))


def lab_to_lin(lab, chroma, rescale):
    l, a, b = lab
    if rescale:
        l = L_BLACK + l * (100 - L_BLACK) / 100
    xyz = lab_to_xyz((l, a / chroma, b / chroma))
    lin = mul(M_INV, tuple((xyz[i] - K * D50[i]) / (1 - K) for i in range(3)))
    mean = sum(lin) / 3
    lin = [(v - (1 - S) * mean) / S for v in lin]
    return tuple(min(1.0, max(0.0, v)) for v in lin)


SAMPLER16 = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_uint16),
    ctypes.POINTER(ctypes.c_uint16),
    ctypes.c_void_p,
)


def clamp16(v):
    return max(0, min(65535, int(round(v * 65535))))


def a2b_sampler(chroma, rescale):
    def sample(inp, out, _):
        lab = lin_to_lab([inp[i] / 65535 for i in range(3)], chroma, rescale)
        out[0] = clamp16(lab[0] / 100)
        out[1] = clamp16((lab[1] + 128) / 255)
        out[2] = clamp16((lab[2] + 128) / 255)
        return 1

    return SAMPLER16(sample)


def b2a_sampler(chroma, rescale):
    def sample(inp, out, _):
        lab = (inp[0] / 65535 * 100, inp[1] / 65535 * 255 - 128, inp[2] / 65535 * 255 - 128)
        lin = lab_to_lin(lab, chroma, rescale)
        for i in range(3):
            out[i] = clamp16(lin[i])
        return 1

    return SAMPLER16(sample)


LCMS.cmsPipelineAlloc.restype = ctypes.c_void_p
LCMS.cmsPipelineAlloc.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint]
LCMS.cmsStageAllocCLut16bit.restype = ctypes.c_void_p
LCMS.cmsStageAllocCLut16bit.argtypes = [
    ctypes.c_void_p,
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_void_p,
]
LCMS.cmsStageAllocToneCurves.restype = ctypes.c_void_p
LCMS.cmsStageAllocToneCurves.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p]
LCMS.cmsStageSampleCLut16bit.argtypes = [ctypes.c_void_p, SAMPLER16, ctypes.c_void_p, ctypes.c_uint]
LCMS.cmsPipelineInsertStage.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
LCMS.cmsCreateProfilePlaceholder.restype = ctypes.c_void_p
LCMS.cmsCreateProfilePlaceholder.argtypes = [ctypes.c_void_p]
LCMS.cmsSetProfileVersion.argtypes = [ctypes.c_void_p, ctypes.c_double]
LCMS.cmsSetDeviceClass.argtypes = [ctypes.c_void_p, ctypes.c_uint]
LCMS.cmsSetColorSpace.argtypes = [ctypes.c_void_p, ctypes.c_uint]
LCMS.cmsSetPCS.argtypes = [ctypes.c_void_p, ctypes.c_uint]
LCMS.cmsWriteTag.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p]
LCMS.cmsSaveProfileToFile.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
LCMS.cmsCloseProfile.argtypes = [ctypes.c_void_p]
LCMS.cmsMLUalloc.restype = ctypes.c_void_p
LCMS.cmsMLUalloc.argtypes = [ctypes.c_void_p, ctypes.c_uint]
LCMS.cmsMLUsetASCII.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
LCMS.cmsBuildTabulatedToneCurveFloat.restype = ctypes.c_void_p
LCMS.cmsBuildTabulatedToneCurveFloat.argtypes = [
    ctypes.c_void_p,
    ctypes.c_uint,
    ctypes.POINTER(ctypes.c_float),
]


def sig(s):
    return int.from_bytes(s.encode(), "big")


class XYZ(ctypes.Structure):
    _fields_ = [("X", ctypes.c_double), ("Y", ctypes.c_double), ("Z", ctypes.c_double)]


def gamma_curves(gamma):
    """Three tone curves `x ** gamma`."""
    LCMS.cmsBuildGamma.restype = ctypes.c_void_p
    LCMS.cmsBuildGamma.argtypes = [ctypes.c_void_p, ctypes.c_double]
    curve = LCMS.cmsBuildGamma(None, gamma)
    curves = (ctypes.c_void_p * 3)(curve, curve, curve)
    return LCMS.cmsStageAllocToneCurves(None, 3, curves)


def clut(grid, sampler, before, after):
    """A pipeline of the curves `before`, a CLUT sampled by `sampler`, the
    curves `after`: the curves on the device side carry the device's gamma,
    so the CLUT holds linear values, as printer profiles do."""
    stage = LCMS.cmsStageAllocCLut16bit(None, grid, 3, 3, None)
    keep.append(sampler)
    if not LCMS.cmsStageSampleCLut16bit(stage, sampler, None, 0):
        sys.exit("sampling failed")
    pipe = LCMS.cmsPipelineAlloc(None, 3, 3)
    # lutAtoBType and lutBtoAType store a CLUT between A and B curves.
    LCMS.cmsPipelineInsertStage(pipe, 1, before)
    LCMS.cmsPipelineInsertStage(pipe, 1, stage)
    LCMS.cmsPipelineInsertStage(pipe, 1, after)
    return pipe


keep = []


def common(profile, version, space, pcs, description):
    LCMS.cmsSetProfileVersion(profile, version)
    LCMS.cmsSetDeviceClass(profile, sig("prtr"))
    LCMS.cmsSetColorSpace(profile, sig(space))
    LCMS.cmsSetPCS(profile, sig(pcs))
    paper = XYZ(*PAPER)
    LCMS.cmsWriteTag(profile, sig("wtpt"), ctypes.byref(paper))
    mlu = LCMS.cmsMLUalloc(None, 1)
    LCMS.cmsMLUsetASCII(mlu, b"en", b"US", description.encode())
    LCMS.cmsWriteTag(profile, sig("desc"), mlu)
    copyright_mlu = LCMS.cmsMLUalloc(None, 1)
    LCMS.cmsMLUsetASCII(copyright_mlu, b"en", b"US", b"No copyright, test data")
    LCMS.cmsWriteTag(profile, sig("cprt"), copyright_mlu)


def rgb_printer(path, version):
    profile = LCMS.cmsCreateProfilePlaceholder(None)
    common(profile, version, "RGB ", "Lab ", f"rasterrocket test RGB printer v{version}")
    for n, chroma, rescale in ((0, 1 / 0.85, True), (1, 1.0, False), (2, 1.1, True)):
        a2b = clut(A2B_GRID, a2b_sampler(chroma, rescale), gamma_curves(2.2), gamma_curves(1.0))
        b2a = clut(B2A_GRID, b2a_sampler(chroma, rescale), gamma_curves(1.0), gamma_curves(1 / 2.2))
        LCMS.cmsWriteTag(profile, sig(f"A2B{n}"), a2b)
        LCMS.cmsWriteTag(profile, sig(f"B2A{n}"), b2a)
    if not LCMS.cmsSaveProfileToFile(profile, path.encode()):
        sys.exit(f"cannot write {path}")
    LCMS.cmsCloseProfile(profile)


def gray_printer(path):
    profile = LCMS.cmsCreateProfilePlaceholder(None)
    common(profile, 4.3, "GRAY", "XYZ ", "rasterrocket test gray printer")
    n = 1024
    table = (ctypes.c_float * n)(*[K + (1 - K) * (i / (n - 1)) ** 1.8 for i in range(n)])
    curve = LCMS.cmsBuildTabulatedToneCurveFloat(None, n, table)
    LCMS.cmsWriteTag(profile, sig("kTRC"), curve)
    if not LCMS.cmsSaveProfileToFile(profile, path.encode()):
        sys.exit(f"cannot write {path}")
    LCMS.cmsCloseProfile(profile)


def main():
    os.makedirs(OUT, exist_ok=True)
    rgb_printer(os.path.join(OUT, "printer-rgb-v4.icc"), 4.3)
    rgb_printer(os.path.join(OUT, "printer-rgb-v2.icc"), 2.4)
    gray_printer(os.path.join(OUT, "printer-gray-v4.icc"))
    assert math.isclose(lab_to_device(device_to_lab((0.3, 0.6, 0.9), 1.0), 1.0)[1], 0.6, rel_tol=1e-9)


if __name__ == "__main__":
    main()
