#!/usr/bin/env python

'''
This module contains some common routines based on OpenCV
'''

import matplotlib.pyplot as plt
import numpy as np

def add_subplot_axes(ax, rect, axisbg='w'):
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]  # <= Typo was here
    subax = fig.add_axes([x, y, width, height], axisbg=axisbg)
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2] ** 0.5
    y_labelsize *= rect[3] ** 0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax

# Visvis related:
shot_counter = 0
filename_prefix = "screenshot_"
file_format = "png"

def OnKey(event):
    """ Called when a key is pressed down in the axes.
    """
    import visvis as vv
    global shot_counter, filename_prefix, file_format

    if event.text and event.text.lower() in 's':
        current_filename = "%s-%02d.%s" % (filename_prefix, shot_counter, file_format)
        print("Saving screenshot as %s" % (current_filename))
        vv.screenshot(filename=current_filename, bg="w", sf=3, format=file_format)
        shot_counter += 1

def draw_axes_visvis(length, line_thickness=1, line_style="-", z_origin_offset=0):
    '''
    @note: We are using Line directly, but it doesn't work with transparency as opposed to using the "solidLine" function generator (They both have their pros and cons)
    @param line_style: Possible line styles (ls) are:
          * Solid line: '-'
          * Dotted line: ':'
          * Dashed line: '--'
          * Dash-dot line: '-.' or '.-'
          * A line that is drawn between each pair of points: '+'
          * No line: '' or None.
    '''
    # make simple, bare axis lines through space:
    import visvis as vv

    a = vv.gca()
    pp_x = vv.Pointset(3)
    pp_x.append(0, 0, z_origin_offset); pp_x.append(length, 0, z_origin_offset);
#     line_x = vv.solidLine(pp_x, radius=line_thickness)
#     line_x.faceColor = "r"
    line_x = vv.Line(a, pp_x)
    line_x.ls = line_style
    line_x.lw = line_thickness
    line_x.lc = "r"

    pp_y = vv.Pointset(3)
    pp_y.append(0, 0, z_origin_offset); pp_y.append(0, length, z_origin_offset);
#     line_y = vv.solidLine(pp_y, radius=line_thickness)
#     line_y.faceColor = "g"
    line_y = vv.Line(a, pp_y)
    line_y.ls = line_style
    line_y.lw = line_thickness
    line_y.lc = "g"

    pp_z = vv.Pointset(3)
    pp_z.append(0, 0, z_origin_offset); pp_z.append(0, 0, z_origin_offset + length);
#     line_z = vv.solidLine(pp_z, radius=line_thickness)
#     line_z.faceColor = "b"
    line_z = vv.Line(a, pp_z)
    line_z.ls = line_style
    line_z.lw = line_thickness
    line_z.lc = "b"


def get_plane_surface(width, height, z_offset, img_face=None):
    import visvis as vv

    xx, yy = np.meshgrid((-width / 2.0, width / 2.0), (-height / 2.0, height / 2.0))
    zz = z_offset + np.zeros_like(xx)

    if img_face:
        plane_surf = vv.surf(xx, yy, zz, img_face)
    else:
        plane_surf = vv.surf(xx, yy, zz)

    return plane_surf

def plot_marginalized_pdf(mean, sigma, n_samples, how_many_sigmas, ax_pdf, x_axis_symbol="x", units="mm"):
    from common_tools import pdf
    data_X = np.linspace(-how_many_sigmas * sigma, how_many_sigmas * sigma, n_samples) + mean
    cons_X = 1. / (np.sqrt(2 * np.pi) * sigma)
    pdf_X = pdf(point=data_X, cons=cons_X, mean=mean, det_sigma=sigma)
    ax_pdf.set_xlabel(r'$%s\,[\mathrm{%s}]$' % (x_axis_symbol, units))
    ax_pdf.set_ylabel(r'$\mathrm{f}_{\mu,\sigma^2}(%s)$' % (x_axis_symbol))
    ax_pdf.plot(data_X, pdf_X,);
    ax_pdf.axvline(mean, color='black', linestyle='--', label="$\mu_{\mathrm{f}_{%s}}=%0.2f \, \mathrm{%s}$" % (x_axis_symbol, mean, units))
    for s in range(how_many_sigmas):
        ax_pdf.axvline(mean + s * sigma, color='red', linestyle=':')  # , label="$%d\sigma_{\mathrm{f}_{%s}}=%0.2f$" % (s, x_axis_symbol))
        ax_pdf.axvline(mean - s * sigma, color='red', linestyle=':')  # , label="$-%d\sigma_{\mathrm{f}_{%s}}=%0.2f$" % (s, x_axis_symbol))

    ax_pdf.legend().draggable()
