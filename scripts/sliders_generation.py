import plotly.graph_objects as go
import base64


class SlidersGeneration():
    
    def __init__(self, image_paths, labels, graph, output_folder):
        self.image_paths = image_paths
        self.labels = labels
        self.graph = graph
        self.output_folder = output_folder
        
        self.init()
        
    def init(self):
        self.combine_images_with_animation()
        

    @staticmethod
    def encode_image(image_path):
        with open(image_path, "rb") as f:
            return "data:image/png;base64," + base64.b64encode(f.read()).decode("utf-8")


    def combine_images_with_animation(self):
        """
        Combines multiple images into an interactive plot with animation_frame.
    
        Parameters
        ----------
        image_paths : list of str
            Paths to the saved image files.
        labels : list of str
            Labels corresponding to each image for the animation.
    
        Returns
        -------
        None
        """
        encoded_images = [self.encode_image(path) for path in self.image_paths]
    
        fig = go.Figure()
    
        # Aggiungi ogni immagine come layout separato
        for i, encoded_img in enumerate(encoded_images):
            fig.add_layout_image(
                source=encoded_img,
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                xanchor="center",
                yanchor="middle",
                sizex=1,
                sizey=1,
                layer="below",
                visible=(i == 0),  # Solo la prima immagine è visibile all'inizio
            )
    
        # Creazione dei frame per l'animazione
        frames = []
        for i, label in enumerate(self.labels):
            frame = go.Frame(
                data=[],
                name=label,
                layout=dict(
                    images=[
                        dict(
                            source=encoded_images[i],
                            xref="paper",
                            yref="paper",
                            x=0.5,
                            y=0.5,
                            xanchor="center",
                            yanchor="middle",
                            sizex=1,
                            sizey=1,
                            layer="below",
                        )
                    ]
                )
            )
            frames.append(frame)
    
        # Aggiungi i frame alla figura
        fig.frames = frames
    
        # Configurazione dello slider
        steps = []
        for i, label in enumerate(self.labels):
            step = dict(
                method="animate",
                args=[
                    [label],
                    dict(frame=dict(duration=10000, redraw=True), mode="immediate"),
                ],
                label=label,
            )
            steps.append(step)
    
        sliders = [dict(
            active=0,
            currentvalue={"prefix": "Image: "},
            pad={"t": 50},
            steps=steps,
        )]
    
        # Configura il layout
        fig.update_layout(
            sliders=sliders,
            title="Combined Images with Animation",
            xaxis=dict(visible=False),  # Nasconde asse X
            yaxis=dict(visible=False),  # Nasconde asse Y
            margin=dict(t=50, b=50, l=50, r=50),
            width=800,
            height=600,
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    buttons=[
                        dict(label="Play", method="animate", args=[None]),
                        dict(label="Pause", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]),
                    ],
                )
            ]
        )
    
        # Visualizza e salva
        fig.show()
        output_name = self.graph
        
        fig.write_html(f"{self.output_folder}/{output_name}.html", auto_play=False)


if __name__ == '__main__':
    # Percorsi delle immagini salvate
    image_paths = [
        "C:\\Users\\aless\\Desktop\\PhD_Pisa\\2025_01_01\\network_analysis\\results\\base_s_39_elec_lvopt_1h.nc\\dispatch_2013-05-01.png",
        "C:\\Users\\aless\\Desktop\\PhD_Pisa\\2025_01_01\\network_analysis\\results\\base_s_39_elec_lvopt_2h.nc\\dispatch_2013-05-01.png",
        "C:\\Users\\aless\\Desktop\\PhD_Pisa\\2025_01_01\\network_analysis\\results\\base_s_39_elec_lvopt_3h.nc\\dispatch_2013-05-01.png",
        "C:\\Users\\aless\\Desktop\\PhD_Pisa\\2025_01_01\\network_analysis\\results\\base_s_39_elec_lvopt_4h.nc\\dispatch_2013-05-01.png",
    ]
    
    # Etichette per l'animazione
    labels = ["Scenario 1", "Scenario 2", "Scenario 3", "Scenario 4"]
    graph = 'dispatch_2013-05-01'
    
    # Crea l'animazione
    sliders_generation = SlidersGeneration(image_paths, labels, graph)
