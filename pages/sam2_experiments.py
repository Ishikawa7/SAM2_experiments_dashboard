import dash
from dash import Dash, Input, Output, State, callback, html, dcc, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px

from PIL import Image
import random
import glob

#from sam2 import load_model
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.utils.misc import variant_to_config_mapping

import numpy as np

dash.register_page(
    __name__,
    path='/',
    )
# load SAM2 model (CPU version)
model = build_sam2(
    variant_to_config_mapping["tiny"],
    "sam2_hiera_tiny.pt",
)
image_predictor = SAM2ImagePredictor(model)
# get the list of all images names in the folder
images_names = glob.glob("data/Kvasir-SEG-processed/train/images/*")
images_names = [ name.split("/")[-1].split(".")[0] for name in images_names]

def create_app_layout():
    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    [
                                        html.Datalist(id="input-list-images-names", children=[html.Option(value=str(name)) for name in images_names]),
                                        dbc.Input(id="input-name", type="text", placeholder="Search for image", list="input-list-images-names", autocomplete="off", debounce=True, style={"width": "25%"}),
                                        dbc.Button("Random image", color="primary", id="random-image-button", style={"margin-left": "15px"}),
                                        dbc.Button("Segment (SAM2)", color="info", id="sam2-button", style={"margin-left": "15px"}),
                                        dbc.RadioItems(
                                            options=[
                                                {"label": "points", "value":"points", "disabled": True},
                                                {"label": "drawrect", "value": "drawrect"},
                                                {"label": "drawclosedpath", "value": "drawclosedpath", "disabled": True},
                                            ],
                                            value='drawrect',
                                            id="radioitems-input-drawmode",
                                            inline=True,
                                            style={"margin-left": "15px"},
                                        ),
                                    ],
                                    style={"display": "flex", "justify-content": "center"},
                                ),
                                dbc.CardBody(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    dbc.Container(
                                                        dcc.Graph(figure=px.scatter(template="plotly_white",height=400, width=400), id="target-mask"),
                                                        style={'justify-content': 'center', 'align-items': 'center', 'display': 'flex'},
                                                    ),
                                                    width=4,
                                                    style={'justify-content': 'center', 'align-items': 'center', 'display': 'flex'},
                                                ),
                                                dbc.Col(
                                                    dbc.Container(
                                                        dcc.Graph(figure=px.scatter(template="plotly_white",height=400, width=400), id="original-image"),
                                                        style={'justify-content': 'center', 'align-items': 'center', 'display': 'flex'},
                                                    ),
                                                    width=4,
                                                    style={'justify-content': 'center', 'align-items': 'center', 'display': 'flex'},
                                                ),
                                                dbc.Col(
                                                    dbc.Container(
                                                        [
                                                            dcc.Graph(figure=px.scatter(template="plotly_white",height=400, width=400)),
                                                        ],
                                                        id = "output-mask-container",
                                                        style={'justify-content': 'center', 'align-items': 'center', 'display': 'flex'},
                                                    ),
                                                    width=4,
                                                    style={'justify-content': 'center', 'align-items': 'center', 'display': 'flex'},
                                                ),
                                            ],
                                            # space between the columns 0
                                            style={"margin": "0"},
                                        ),                                       
                                    ]
                                ),
                                #dbc.CardFooter("This is the footer"),
                            ],
                        ),
                    ),
                ],
            ),
            html.Br(),
        ],
        fluid=True,
    )

layout = create_app_layout

# create callback that read the image and display it
@callback(
    Output("original-image", "figure"),
    Output("target-mask", "figure"),
    Output("input-name", "value"),
    Output("original-image", "relayoutData", allow_duplicate=True),
    Output("output-mask-container", "children", allow_duplicate=True),
    Input("input-name", "value"),
    Input("random-image-button", "n_clicks"),
    Input("radioitems-input-drawmode", "value"),
    prevent_initial_call=True,
)
def display_image(input_name, n_clicks, drawmode):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate
    else:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if button_id == "random-image-button":
            input_name = random.choice(images_names)
        elif input_name not in images_names:
            raise dash.exceptions.PreventUpdate
        img = Image.open(f"data/Kvasir-SEG-processed/train/images/{input_name}.jpg")
        mask = Image.open(f"data/Kvasir-SEG-processed/train/masks/{input_name}.jpg")
        img = Image.open(f"data/Kvasir-SEG-processed/train/images/{input_name}.jpg")
        mask = Image.open(f"data/Kvasir-SEG-processed/train/masks/{input_name}.jpg")
        fig_original = px.imshow(img)
        fig_original.update_layout( dragmode=drawmode, newshape=dict(line_color='cyan'), title="Original image",title_x=0.5)
        # remove margins
        fig_original.update_layout(height=400, width=400)#, margin=dict(l=0, r=0, b=0, t=0))
        fig_target = px.imshow(mask)
        fig_target.update_layout(coloraxis_showscale=False, title="Target",title_x=0.5)
        fig_target.update_layout(height=400, width=400)#, margin=dict(l=0, r=0, b=0, t=0))
        # display option drawclosedpath, drawrect, drawline in the fig_original
        return fig_original, fig_target, input_name, None, dcc.Graph(figure=px.scatter(template="plotly_white",height=400, width=400)) #.show(config={'modeBarButtonsToAdd':['drawclosedpath','drawrect','eraseshape']})
    
# create a callback that set up a spinner while the model is predicting
@callback(
    Output("output-mask-container", "children", allow_duplicate=True),
    Input("sam2-button", "n_clicks"),
    State("original-image", "relayoutData"),
    prevent_initial_call=True,
)
def set_spinner(n_clicks, relayoutData):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate
    else:
        # check if relayoutData has key 'shapes'
        if relayoutData is None or "shapes" not in relayoutData:
            return [dbc.Alert("Please select an area", color="warning")]
        return [dbc.Spinner(color="primary", type="grow", size="lg")]

# create callback that get the selected area and display it in the selection graph
@callback(
    Output("output-mask-container", "children", allow_duplicate=True),
    Output("original-image", "relayoutData", allow_duplicate=True),
    Input("output-mask-container", "children"),
    State("original-image", "relayoutData"),
    State("input-name", "value"),
    State("output-mask-container", "children"),
    prevent_initial_call=True,
)
def display_selection(n_clicks, relayoutData, image_name, actual_children):
    ctx = dash.callback_context
    if not ctx.triggered  or image_name not in images_names or relayoutData is None or "shapes" not in relayoutData:
        raise dash.exceptions.PreventUpdate
    else:
        if actual_children == [] or actual_children == None:
            return [dcc.Graph(figure=px.scatter(template="plotly_white",height=400, width=400))], None
        x0 = int(relayoutData['shapes'][-1]["x0"])
        x1 = int(relayoutData['shapes'][-1]["x1"])
        y0 = int(relayoutData['shapes'][-1]["y0"])
        y1 = int(relayoutData['shapes'][-1]["y1"])
        box = [x0, y0, x1, y1]
        # get the image and convert it to numpy array
        img = np.array(Image.open(f"data/Kvasir-SEG-processed/train/images/{image_name}.jpg").convert("RGB"))

        image_predictor.set_image(img)
        predicted_mask, _, _ = image_predictor.predict(box=box, multimask_output=False)
        fig = px.imshow(predicted_mask[0])
        # hide the colorbar
        fig.update_layout(coloraxis_showscale=False, title="Predicted",title_x=0.5)
        fig.update_layout(height=400, width=400)
        return [dcc.Graph(figure=fig)], None