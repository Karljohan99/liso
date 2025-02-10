import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import numpy as np

from visualize_mined_dbs import get_scene_elements as get_mined_dbs_elements
from visualize_scene_flow import get_scene_elements as get_scene_flow_elements

class PointCloudApp:
    def __init__(self):
        self.app = gui.Application.instance
        self.app.initialize()

        # Create the window
        self.window = self.app.create_window("Point Cloud Viewer", 1024, 768)

        # Add keybindings
        self.window.set_on_key(self.on_key)

        # Create 3D Scene
        self.scene = gui.SceneWidget()
        self.window.add_child(self.scene)

        # Initialize Open3D Scene
        self.scene.scene = rendering.Open3DScene(self.window.renderer)

        self.mode = "mined_dbs"

        self.material = rendering.MaterialRecord()

        self.material.point_size = 3.0
        self.bbox_len = 0
        self.arrow_len = 0
        self.index = "000"

        # Setup the camera
        self.scene.setup_camera(60, o3d.geometry.AxisAlignedBoundingBox(np.array([-100.0]*3), np.array([100.0]*3)), np.zeros((3, 1)))

        # Initialize buttons and text fileds
        self.dataset_textbox = gui.TextEdit()
        self.dataset_textbox.placeholder_text = "Enter dataset"
        self.dataset_textbox.set_on_value_changed(self.on_dataset_changed)

        self.sequence_textbox = gui.TextEdit()
        self.sequence_textbox.placeholder_text = "Enter sequence"
        self.sequence_textbox.set_on_value_changed(self.on_sequence_changed)

        self.index_textbox = gui.TextEdit()
        self.index_textbox.placeholder_text = "Seq Idx"
        self.index_textbox.set_on_value_changed(self.on_index_changed)

        self.ok_button = gui.Button("OK")
        self.ok_button.set_on_clicked(self.visualize_point_cloud)

        self.back_button = gui.Button("<--")
        self.back_button.set_on_clicked(self.go_backward)

        self.forward_button = gui.Button("-->")
        self.forward_button.set_on_clicked(self.go_forward)

        self.switch_mode_button = gui.Button("Mined boxes")
        self.switch_mode_button.set_on_clicked(self.switch_mode)

        # Add components to window
        self.window.add_child(self.dataset_textbox)
        self.window.add_child(self.sequence_textbox)
        self.window.add_child(self.index_textbox)
        self.window.add_child(self.ok_button)
        self.window.add_child(self.back_button)
        self.window.add_child(self.forward_button)
        self.window.add_child(self.switch_mode_button)

        # Set layout
        self.window.set_on_layout(self.on_layout)

    def on_key(self, event):
        # Check for key press events
        if event.type == gui.KeyEvent.DOWN:
            if event.key == gui.KeyName.LEFT:
                self.go_backward()
            elif event.key == gui.KeyName.RIGHT:
                self.go_forward()
            elif event.key == gui.KeyName.ENTER:
                self.visualize_point_cloud()

    def on_dataset_changed(self, new_text):
        self.dataset = new_text.strip()

    def on_sequence_changed(self, new_text):
        self.sequence = new_text.strip()

    def on_index_changed(self, new_text):
        self.index = new_text.strip().zfill(3)

    def go_backward(self):
        new_index = str(int(self.index) - 1)
        self.index_textbox.text_value = new_index
        self.index = new_index.zfill(3)
        self.visualize_point_cloud()

    def go_forward(self):
        new_index = str(int(self.index) + 1)
        self.index_textbox.text_value = new_index
        self.index = new_index.zfill(3)
        self.visualize_point_cloud()

    def switch_mode(self):
        if self.mode == "mined_dbs":
            self.mode = "scene_flow"
            self.switch_mode_button.text = "Scene flow"

        elif self.mode == "scene_flow":
            self.mode = "mined_dbs"
            self.switch_mode_button.text = "Mined boxes"

    def visualize_point_cloud(self):

        try:
            if self.mode == "mined_dbs":
                pcd, boxes = get_mined_dbs_elements(self.dataset, self.index)
                
                # Remove and re-add the geometry to apply changes
                self.scene.scene.remove_geometry("pointcloud")
                self.scene.scene.add_geometry("pointcloud", pcd, self.material)

                self.scene.scene.remove_geometry("arrows")

                for i in range(self.bbox_len):
                    self.scene.scene.remove_geometry(f"bbox_{i}")

                # Add bounding boxes to the scene
                self.bbox_len = len(boxes)
                for i in range(self.bbox_len):
                    self.scene.scene.add_geometry(f"bbox_{i}", boxes[i], self.material)

            elif self.mode == "scene_flow":
                pcd, arrows = get_scene_flow_elements(self.dataset, self.sequence, self.index)
                
                # Remove and re-add the geometry to apply changes
                self.scene.scene.remove_geometry("pointcloud")
                self.scene.scene.add_geometry("pointcloud", pcd, self.material)

                for i in range(self.bbox_len):
                    self.scene.scene.remove_geometry(f"bbox_{i}")

                self.scene.scene.remove_geometry("arrows")

                # Merge all arrows into a single mesh
                merged_arrows = o3d.geometry.TriangleMesh()
                for arrow in arrows:
                    merged_arrows += arrow

                # Add arrow mesh to the scene
                self.scene.scene.add_geometry("arrows", merged_arrows, self.material)

        except Exception as e:
            print(repr(e))


    def on_layout(self, layout_context):
        """Set UI layout."""
        self.scene.frame = self.window.content_rect
        self.dataset_textbox.frame = gui.Rect(self.window.content_rect.x + 10, 
                                            self.window.content_rect.get_bottom() - 50, 
                                            150, 30)
        self.sequence_textbox.frame = gui.Rect(self.window.content_rect.x + 170, 
                                            self.window.content_rect.get_bottom() - 50, 
                                            150, 30)
        
        self.back_button.frame = gui.Rect(self.window.content_rect.x + 330, 
                                            self.window.content_rect.get_bottom() - 50, 
                                            50, 30)
        
        self.index_textbox.frame = gui.Rect(self.window.content_rect.x + 390, 
                                            self.window.content_rect.get_bottom() - 50, 
                                            70, 30)
        
        self.forward_button.frame = gui.Rect(self.window.content_rect.x + 470, 
                                            self.window.content_rect.get_bottom() - 50, 
                                            50, 30)
                                            
        self.ok_button.frame = gui.Rect(self.window.content_rect.x + 550, 
                                           self.window.content_rect.get_bottom() - 50, 
                                           50, 30)
        
        self.switch_mode_button.frame = gui.Rect(self.window.content_rect.x + 10, 
                                           10, 150, 30)

    def run(self):
        self.app.run()

# Run the application
if __name__ == "__main__":
    PointCloudApp().run()
