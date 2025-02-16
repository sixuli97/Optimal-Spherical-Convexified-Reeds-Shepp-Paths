# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 10:17:19 2022

@author: deepak
"""

import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "browser"
from plotly.figure_factory import create_quiver


class plotting_functions:

    def __init__(self):

        # Declaring as plotly graphic objects figure environment
        self.fig = go.Figure()

    # Function for adding a trace
    def add_trace(self, data):
        self.fig.add_traces(data)

    # Function for adding a 2D scatter plot
    def scatter_2D(
        self, xdata, ydata, color_plot, legend, linewidth=3, linestyle="solid"
    ):
        self.fig.add_trace(
            go.Scatter(
                x=xdata,
                y=ydata,
                mode="lines",
                line=dict(color=color_plot, width=linewidth, dash=linestyle),
                name=legend,
                showlegend=legend is not None,
            )
        )

    # Function for adding points to 2D plot
    def points_2D(
        self, xcoord, ycoord, color_marker, legend, marker_type="circle", marker_size=8
    ):
        self.fig.add_trace(
            go.Scatter(
                x=xcoord,
                y=ycoord,
                mode="markers",
                marker=dict(
                    size=marker_size,
                    colorscale="Viridis",
                    symbol=marker_type,
                    color=color_marker,
                ),
                name=legend,
                showlegend=legend is not None,
            )
        )

    # Function for adding arrows to 2D plot
    def arrows_2D(
        self,
        xstart,
        ystart,
        dir_cos_x,
        dir_cos_y,
        color_arrow,
        legend,
        arrowsize=3,
        linewidth=3,
    ):
        temp = create_quiver(
            x=xstart,
            y=ystart,
            u=dir_cos_x,
            v=dir_cos_y,
            line=dict(width=linewidth, color=color_arrow),
            scale=arrowsize,
            name=legend,
            showlegend=legend is not None,
        )
        self.fig.add_traces(data=temp.data)

    # Function for adding a surface to 3D plot
    def surface_3D(self, xdata, ydata, zdata, color, legend, opacity_surf):
        self.fig.add_trace(
            go.Surface(
                x=xdata,
                y=ydata,
                z=zdata,
                name=legend,
                showscale=False,
                colorscale=[[0, color], [1, color]],
                opacity=opacity_surf,
                showlegend=legend is not None,
            )
        )

    # Function for adding a 3D scatter plot
    def scatter_3D(
        self,
        xdata,
        ydata,
        zdata,
        color_plot,
        legend,
        linewidth=6,
        linestyle="solid",
        text="n",
        text_str="none",
        text_position="top right",
        text_size=22,
    ):
        self.fig.add_trace(
            go.Scatter3d(
                x=xdata,
                y=ydata,
                z=zdata,
                mode="lines" + ("+text" if text.lower() == "y" else ""),
                line=dict(color=color_plot, width=linewidth, dash=linestyle),
                name=legend,
                text=["", text_str] if text.lower() == "y" else None,
                textposition=text_position,
                textfont_size=text_size,
                showlegend=legend is not None,
            )
        )

    # Function for adding points to 3D plot
    def points_3D(
        self,
        xcoord,
        ycoord,
        zcoord,
        color_marker,
        legend,
        marker_type="circle",
        marker_size=6,
    ):
        self.fig.add_trace(
            go.Scatter3d(
                x=xcoord,
                y=ycoord,
                z=zcoord,
                mode="markers",
                marker=dict(
                    size=marker_size,
                    colorscale="Viridis",
                    symbol=marker_type,
                    color=color_marker,
                ),
                name=legend,
                showlegend=legend is not None,
            )
        )

    # Function for adding arrow to 3D plot
    def arrows_3D(
        self,
        xstart,
        ystart,
        zstart,
        dir_cos_x,
        dir_cos_y,
        dir_cos_z,
        color_arrow_line,
        color_arrow_tip,
        legend,
        linewidth=3,
        arrowsize=2,
        arrowtipsize=2,
        text="n",
        text_str="none",
        text_position="top right",
        text_size=22,
    ):

        # Adding the line corresponding to the arrow using the scatter_3D function
        self.scatter_3D(
            [xstart[0], xstart[0] + dir_cos_x[0] * arrowsize],
            [ystart[0], ystart[0] + dir_cos_y[0] * arrowsize],
            [zstart[0], zstart[0] + dir_cos_z[0] * arrowsize],
            color_arrow_line,
            legend,
            linewidth,
            "solid",
            text,
            text_str,
            text_position,
            text_size=22,
        )
        # Adding a cone for the end of the arrow
        self.fig.add_trace(
            go.Cone(
                x=[xstart[0] + dir_cos_x[0] * arrowsize],
                y=[ystart[0] + dir_cos_y[0] * arrowsize],
                z=[zstart[0] + dir_cos_z[0] * arrowsize],
                u=[dir_cos_x[0] * arrowtipsize],
                v=[dir_cos_y[0] * arrowtipsize],
                w=[dir_cos_z[0] * arrowtipsize],
                colorscale=color_arrow_tip,
                showscale=False,
                showlegend=False,
            )
        )

    # Updating layout for 2D plot
    def update_layout_2D(self, xlabel, xlim, ylabel, ylim, plot_title):
        self.fig.update_layout(
            xaxis_title="x (m)",
            yaxis_title="z (m)",
            title_text=plot_title,
            xaxis_range=xlim,
            yaxis_range=ylim,
        )

    # Updating layout for 3D plot
    def update_layout_3D(self, xlabel, ylabel, zlabel, plot_title):
        self.fig.update_layout(
            width=1200,
            height=1050,
            # autosize=False,
            scene=dict(
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    eye=dict(
                        # x = 0,
                        # y = 1.0707,
                        # z = 1,
                        x=2.1,
                        y=0.2,
                        z=0.4,
                    ),
                ),
                # aspectratio = dict(x = 0.75, y = 0.75, z = 0.5),
                # aspectmode = 'manual',
                aspectmode="cube",
                xaxis=dict(
                    title=dict(
                        text=xlabel, font=dict(size=26)
                    ),  # Adjust font size for x-axis title
                    tickfont=dict(size=16),  # Adjust font size for x-axis ticks
                    dtick=1,
                    range=[-1.5, 1.5],
                ),
                yaxis=dict(
                    title=dict(
                        text=ylabel, font=dict(size=26)
                    ),  # Adjust font size for y-axis title
                    tickfont=dict(size=16),  # Adjust font size for y-axis ticks
                    dtick=1,
                    range=[-1.5, 1.5],
                ),
                zaxis=dict(
                    title=dict(
                        text=zlabel, font=dict(size=26)
                    ),  # Adjust font size for z-axis title
                    tickfont=dict(size=16),  # Adjust font size for z-axis ticks
                    dtick=1,
                    range=[-1.5, 1.5],
                ),
            ),
            # title_text=plot_title,
            legend=dict(
                x=1.2,  # Adjust x position
                y=0.5,  # Adjust y position
                xanchor="right",
                yanchor="middle",
                font=dict(size=22),  # Set the font size for the legend
                bgcolor="rgba(255, 255, 255, 0.5)",  # Add a semi-transparent background to the legend
                bordercolor="black",  # Set the border color to black
                borderwidth=1,  # Set the border width
                # itemsizing="constant",
                itemwidth=45,
            ),
        )

    def writing_fig_to_html(self, filename, mode="a"):
        with open(filename, mode) as f:
            f.write(self.fig.to_html(full_html=False, include_plotlyjs="cdn"))

    def show(self):
        self.fig.show()
