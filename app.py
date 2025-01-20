import dash
from dash import Dash, Input, Output, State, callback, html, dcc
import dash_bootstrap_components as dbc
import dash_auth

# only for development
VALID_USERNAME_PASSWORD_PAIRS = {
    'labmednetnea': 'zioSAM2',
}

# css file for dash components
dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"
# LOAD DATA #############################################################################################################

# DASH APP ##############################################################################################################
app = Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.SANDSTONE, dbc_css]) #suppress_callback_exceptions=True

auth = dash_auth.BasicAuth(
    app,
    VALID_USERNAME_PASSWORD_PAIRS
)

def create_app_layout():
    return dbc.Container(
        [
            dbc.NavbarSimple(
                id = "navbar",
                children=[
                    #html.Img(src='/static/logo/vrm2.jpg', height="50px", style={"align": "left", "margin-right": "600px"}),
                    dbc.NavItem(dbc.NavLink("HOME", href="/", style={'font-size': '25px'})),
                    dbc.NavItem(dbc.NavLink("Guide", href="/guide", style={'font-size': '25px'})),
                    #dropdown menu with links
                    dbc.DropdownMenu(
                        nav=True,
                        in_navbar=True,
                        label="Pages",
                        children=[
                            dbc.DropdownMenuItem("SAM2 experiments", href="/"),
                            dbc.DropdownMenuItem("Guide", href="/guide"),
                            #dbc.DropdownMenuItem("Dynamic pricing", href="/dynamic_pricing"),
                        ],
                    ),
                    #html.Div(style={'display': 'inline-block', 'width': '95px'}),
                    #html.Img(src='/static/logo/vrm4.jpg', height="40px"),
                ],
                brand="SAM2 manual tests and experiments v0.0.1",
                brand_href="/",
                color="primary",
                dark=True,
                #background image
                #style={"background-image": "url('/static/images/home_ittigain_ok.jpg')", "background-size": "100% 100%", "background-repeat": "no-repeat", "background-position": "center", "height": "250px"},
            ),
            html.Hr(),
    	    dash.page_container,
            # add a footer
            html.Br(),
            html.Hr(),
        ],
        fluid=True,
        style={"background-color": "white"},
    )

app.layout = create_app_layout

# RUN THE APP ###########################################################################################################
if __name__ == "__main__":
    # shut down any running dash processes if necessary
    #import os
    #os.system("taskkill /f /im python.exe")
    
    # start the dash app
    #app.run_server(host='0.0.0.0', port=8080, debug=False, use_reloader=False) # for production
    app.run_server(debug=True, use_reloader=True, port=8080) # for development
    

# if Python [Errno 98] Address already in use: "kill -9 $(ps -A | grep python | awk '{print $1}')"