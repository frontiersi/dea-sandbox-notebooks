# Load modules
from ipyleaflet import (
    Map,
    GeoJSON,
    DrawControl,
    basemaps
)
import datetime as dt
import datacube
import ogr
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import rasterio
import xarray as xr
import IPython
from IPython.display import display
import warnings
import ipywidgets as widgets
from datacube.storage import masking
# Load utility functions
from utils.DEADataHandling import load_clearsentinel2
from utils.utils import transform_from_wgs_poly
from utils.BandIndices import calculate_indices

def load_mining_data():
    """
    Loads Sentinel-2 Near Real Time (NRT) product for the agriculture
    case-study area. The NRT product is provided for the last 90 days.
    Last modified: May 2019
    Author: Caitlin Adams (FrontierSI)

    outputs
    ds - data set containing combined, masked data from Sentinel-2a and -2b.
    Masked values are set to 'nan'
    """
    # Suppress warnings
    warnings.filterwarnings('ignore')

    # Initialise the data cube. 'app' argument is used to identify this app
    dc = datacube.Datacube(app='mining-app')

    # Specify latitude and longitude ranges
    latitude = (-34.426512, -34.434517)
    longitude = (116.648123, 116.630731)

    # Specify the date range
    # Calculated as today's date, subtract 90 days to match NRT availability
    # Dates are converted to strings as required by loading function below
    time = ("2015-01-01", "2018-12-31")

    # Construct the data cube query
    query = {
        'x': longitude,
        'y': latitude,
        'time': time,
        'output_crs': 'EPSG:3577',
        'resolution': (-25, 25)
    }

    print("Loading Fractional Cover")
    dataset_fc = dc.load(
        product="ls8_fc_albers",
        **query,
    )

    print("Loading WoFS")
    dataset_wofs = dc.load(
        product="wofs_albers",
        like=dataset_fc
    )
    
    # Match the data
    shared_times = np.intersect1d(dataset_fc.time, dataset_wofs.time)

    ds_fc_matched = dataset_fc.sel(time=shared_times)
    ds_wofs_matched = dataset_wofs.sel(time=shared_times)
    
    # Mask FC
    dry_mask = masking.make_mask(ds_wofs_matched, dry=True)
    
    # Get fractional masked datasets
    ds_fc_masked = ds_fc_matched.where(dry_mask.water == True)
    ds_wofs_masked = masking.make_mask(ds_wofs_matched, wet=True)
    ds_wofs_masked['water_fraction'] = ds_wofs_masked.water.astype(float)
    ds_fc_masked['ground_cover'] = ds_fc_masked.PV + ds_fc_masked.NPV
    
    # Merge data sets
    print("Combining and resampling data")
    ds_combined = xr.merge([ds_fc_masked/100, ds_wofs_masked.water_fraction])
    
    # Resample
    ds_combined_resampled = ds_combined.resample(time="1M").median()
    ds_combined_resampled.attrs['crs'] = dataset_fc.crs


    # Return the data
    return(ds_combined_resampled)


def run_mining_app(ds):
    """
    Plots an interactive map of the agriculture case-study area and allows
    the user to draw polygons. This returns a plot of the average NDVI value
    in the polygon area.
    Last modified: May 2019
    Author: Caitlin Adams (FrontierSI)

    inputs
    ds - data set containing combined, masked data from Sentinel-2a and -2b.
    Must also have an attribute containing the NDVI value for each pixel
    """
    # Suppress warnings
    warnings.filterwarnings('ignore')

    # Update plotting functionality through rcParams
    mpl.rcParams.update({'figure.autolayout': True})

    # Define the bounding box that will be overlayed on the interactive map
    # The bounds are hard-coded to match those from the loaded data
    geom_obj = {
        "type": "Feature",
        "properties": {
            "style": {
                "stroke": True,
                "color": 'red',
                "weight": 4,
                "opacity": 0.8,
                "fill": True,
                "fillColor": False,
                "fillOpacity": 0,
                "showArea": True,
                "clickable": True
            }
        },
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [
                    [116.630731, -34.434517],
                    [116.630731, -34.426512],
                    [116.648123, -34.426512],
                    [116.648123, -34.434517],
                    [116.630731, -34.434517]
                ]
            ]
        }
    }

    # Create a map geometry from the geom_obj dictionary
    # center specifies where the background map view should focus on
    # zoom specifies how zoomed in the background map should be
    loadeddata_geometry = ogr.CreateGeometryFromJson(str(geom_obj['geometry']))
    loadeddata_center = [
        loadeddata_geometry.Centroid().GetY(),
        loadeddata_geometry.Centroid().GetX()
    ]
    loadeddata_zoom = 15

    # define the study area map
    studyarea_map = Map(
        center=loadeddata_center,
        zoom=loadeddata_zoom,
        basemap=basemaps.Esri.WorldImagery
    )

    # define the drawing controls
    studyarea_drawctrl = DrawControl(
        polygon={"shapeOptions": {"fillOpacity": 0}},
        marker={},
        circle={},
        circlemarker={},
        polyline={},
    )

    # add drawing controls and data bound geometry to the map
    studyarea_map.add_control(studyarea_drawctrl)
    studyarea_map.add_layer(GeoJSON(data=geom_obj))

    # Index to count drawn polygons
    polygon_number = 0

    # Define widgets to interact with
    instruction = widgets.Output(layout={'border': '1px solid black'})
    with instruction:
        print("Draw a polygon within the red box to view a plot of "
              "average NDVI over time in that area.")

    info = widgets.Output(layout={'border': '1px solid black'})
    with info:
        print("Plot status:")

    fig_display = widgets.Output(layout=widgets.Layout(
        width="50%",  # proportion of horizontal space taken by plot
    ))

    with fig_display:
        plt.ioff()
        fig, ax = plt.subplots(3, 1, figsize=(8, 12))
        
        for axis in ax:
            axis.set_ylim([0,1])

    colour_list = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Function to execute each time something is drawn on the map
    def handle_draw(self, action, geo_json):
        nonlocal polygon_number

        # Execute behaviour based on what the user draws
        if geo_json['geometry']['type'] == 'Polygon':

#             info.clear_output(wait=True)  # wait=True reduces flicker effect
#             with info:
#                 print("Plot status: polygon added to plot")

            # Convert the drawn geometry to pixel coordinates
            geom_selectedarea = transform_from_wgs_poly(
                geo_json['geometry'],
                EPSGa=3577  # hard-coded to be same as case-study data
            )
            
#             info.clear_output(wait=True)  # wait=True reduces flicker effect
#             with info:
#                 print(geom_selectedarea)

            # Construct a mask to only select pixels within the drawn polygon
            mask = rasterio.features.geometry_mask(
                [geom_selectedarea for geoms in [geom_selectedarea]],
                out_shape=ds.geobox.shape,
                transform=ds.geobox.affine,
                all_touched=False,
                invert=True
            )
        
#             info.clear_output(wait=True)  # wait=True reduces flicker effect
#             with info:
#                 print("Plot status: mask made")

            masked_ds = ds.where(mask)
            masked_ds_mean = masked_ds.mean(dim=['x', 'y'], skipna=True)
            
            colour = colour_list[polygon_number % len(colour_list)]

            # Add a layer to the map to make the most recently drawn polygon
            # the same colour as the line on the plot
            studyarea_map.add_layer(
                GeoJSON(
                    data=geo_json,
                    style={
                        'color': colour,
                        'opacity': 1,
                        'weight': 4.5,
                        'fillOpacity': 0.0
                    }
                )
            )

            # add new data to the plot
#             xr.plot.plot(
#                 masked_ds_mean.PV,
#                 marker='*',
#                 color=colour,
#                 ax=ax[0]
#             )
            masked_ds_mean.BS.interpolate_na(dim="time", method="nearest").plot.line('-', ax=ax[0])
            masked_ds_mean.PV.interpolate_na(dim="time", method="nearest").plot.line('-', ax=ax[1])
            masked_ds_mean.NPV.interpolate_na(dim="time", method="nearest").plot.line('-', ax=ax[2])
            #masked_ds_mean.water_fraction.interpolate_na(dim="time", method="nearest").plot('-', ax=ax[3])
            
#             xr.plot.plot(
#                 masked_ds_mean.NPV,
#                 marker='*',
#                 color=colour,
#                 ax=ax[1]
#             )

            # reset titles back to custom
            ax[0].set_ylabel("Bare soil")
            ax[1].set_ylabel("Green vegetation")
            ax[2].set_ylabel("Brown vegetation")
#             ax[3].set_ylabel("Water")

            # refresh display
            fig_display.clear_output(wait=True)  # wait=True reduces flicker effect
            with fig_display:
                display(fig)

            # Iterate the polygon number before drawing another polygon
            polygon_number = polygon_number + 1

        else:
            info.clear_output(wait=True)
            with info:
                print("Plot status: this drawing tool is not currently "
                      "supported. Please use the polygon tool.")

    # call to say activate handle_draw function on draw
    studyarea_drawctrl.on_draw(handle_draw)

    with fig_display:
        # TODO: update with user friendly something
        display(widgets.HTML(""))

    # Construct UI:
    #  +-----------------------+
    #  | instruction           |
    #  +-----------+-----------+
    #  |  map      |  plot     |
    #  |           |           |
    #  +-----------+-----------+
    #  | info                  |
    #  +-----------------------+
    ui = widgets.VBox([instruction,
                       widgets.HBox([studyarea_map, fig_display]),
                       info])
    display(ui)