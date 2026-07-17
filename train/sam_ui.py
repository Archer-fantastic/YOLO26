import cv2
from sam import SAMInferencer


class SAMUI:
    def __init__(self, img_path=r"D:\Min\Projects\VSCodeProjects\ultralytics-main\train\test_imgs\img1.jpg"):
        self.img_path = img_path
        self.inferencer = SAMInferencer()

        # Load the image
        self.image = cv2.imread(img_path)
        self.image_copy = self.image.copy()
        self.temp_image = self.image.copy()

        # Prompt mode: 0=points, 1=rectangle, 2=text
        self.prompt_mode = 0

        # Points and labels storage
        self.points = []
        self.labels = []

        # Rectangle storage
        self.rect_start = (-1, -1)
        self.rect_end = (-1, -1)
        self.drawing_rect = False

        # Text prompt storage
        self.text_prompt = ""

        # Colors
        self.positive_color = (0, 255, 0)  # Green
        self.negative_color = (0, 0, 255)  # Red
        self.rect_color = (255, 255, 0)  # Cyan
        self.text_color = (255, 255, 255)  # White
        self.bg_color = (0, 0, 0)  # Black

        # Create window and set mouse callback
        self.window_name = "SAM Interactive UI"
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        # Display instructions
        self.display_instructions()

    def display_instructions(self):
        """Display instructions on how to use the UI."""
        print("=== SAM Interactive UI Instructions ===")
        print("Mode Selection:")
        print("  0 - Points Mode: Left click for positive points, right click for negative points")
        print("  1 - Rectangle Mode: Drag to draw a bounding box")
        print("  2 - Text Mode: Enter text prompt")
        print("Keys:")
        print("  0, 1, 2 - Switch between modes")
        print("  Enter - Run inference")
        print("  c - Clear all prompts")
        print("  s - Save current results")
        print("  q - Quit")
        print("========================================")

    def draw_ui_elements(self):
        """Draw UI elements on the image."""
        # Create a copy of the image with instructions
        display_img = self.temp_image.copy()

        # Draw mode indicator
        mode_text = f"Mode: {['Points', 'Rectangle', 'Text'][self.prompt_mode]}"
        cv2.putText(display_img, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.text_color, 2)

        # Draw points
        for i, (px, py) in enumerate(self.points):
            color = self.positive_color if self.labels[i] == 1 else self.negative_color
            label = f"+{i}" if self.labels[i] == 1 else f"-{i}"
            cv2.circle(display_img, (px, py), 5, color, -1)
            cv2.putText(display_img, label, (px + 10, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw rectangle if in progress or completed
        if self.drawing_rect and self.rect_start != (-1, -1):
            cv2.rectangle(
                display_img, self.rect_start, (self.current_mouse_x, self.current_mouse_y), self.rect_color, 2
            )
        elif self.rect_start != (-1, -1) and self.rect_end != (-1, -1):
            cv2.rectangle(display_img, self.rect_start, self.rect_end, self.rect_color, 2)

        # Draw text prompt if any
        if self.text_prompt:
            cv2.putText(
                display_img, f"Text: {self.text_prompt}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.text_color, 2
            )

        return display_img

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events."""
        self.current_mouse_x, self.current_mouse_y = x, y

        if self.prompt_mode == 0:  # Points mode
            if event == cv2.EVENT_LBUTTONDOWN:
                # Add positive point
                self.points.append([x, y])
                self.labels.append(1)
                self.temp_image = self.image.copy()
            elif event == cv2.EVENT_RBUTTONDOWN:
                # Add negative point
                self.points.append([x, y])
                self.labels.append(0)
                self.temp_image = self.image.copy()

        elif self.prompt_mode == 1:  # Rectangle mode
            if event == cv2.EVENT_LBUTTONDOWN:
                # Start drawing rectangle
                self.drawing_rect = True
                self.rect_start = (x, y)
            elif event == cv2.EVENT_MOUSEMOVE:
                # Update rectangle while drawing
                if self.drawing_rect:
                    self.temp_image = self.image.copy()
            elif event == cv2.EVENT_LBUTTONUP:
                # Finish drawing rectangle
                self.drawing_rect = False
                self.rect_end = (x, y)
                self.temp_image = self.image.copy()

    def run(self):
        """Main loop for the UI."""
        while True:
            # Draw UI elements
            display_img = self.draw_ui_elements()

            # Show the image
            cv2.imshow(self.window_name, display_img)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == ord("Q"):
                # Quit
                cv2.destroyAllWindows()
                break

            elif key == ord("c") or key == ord("C"):
                # Clear all prompts
                self.clear_prompts()

            elif key == ord("s") or key == ord("S"):
                # Save current results
                self.save_results()

            elif key == ord("0"):
                # Switch to points mode
                self.prompt_mode = 0
                print("Switched to Points Mode")

            elif key == ord("1"):
                # Switch to rectangle mode
                self.prompt_mode = 1
                print("Switched to Rectangle Mode")

            elif key == ord("2"):
                # Switch to text mode
                self.prompt_mode = 2
                print("Switched to Text Mode")
                self.enter_text_prompt()

            elif key == 13:  # Enter key
                # Run inference
                self.run_inference()

    def clear_prompts(self):
        """Clear all prompts."""
        self.points = []
        self.labels = []
        self.rect_start = (-1, -1)
        self.rect_end = (-1, -1)
        self.drawing_rect = False
        self.text_prompt = ""
        self.image = self.image_copy.copy()
        self.temp_image = self.image_copy.copy()
        print("All prompts cleared")

    def enter_text_prompt(self):
        """Enter text prompt via keyboard."""
        self.text_prompt = input("Enter text prompt: ")
        print(f"Text prompt set to: {self.text_prompt}")

    def run_inference(self):
        """Run inference based on the current prompt mode."""
        print("Running inference...")

        if self.prompt_mode == 0:  # Points mode
            if len(self.points) > 0:
                results = self.inferencer.infer(self.img_path, points=self.points, labels=self.labels)
            else:
                print("No points added. Please add at least one point.")
                return

        elif self.prompt_mode == 1:  # Rectangle mode
            if self.rect_start != (-1, -1) and self.rect_end != (-1, -1):
                # Ensure rectangle coordinates are in order
                x1 = min(self.rect_start[0], self.rect_end[0])
                y1 = min(self.rect_start[1], self.rect_end[1])
                x2 = max(self.rect_start[0], self.rect_end[0])
                y2 = max(self.rect_start[1], self.rect_end[1])
                bboxes = [x1, y1, x2, y2]
                results = self.inferencer.infer(self.img_path, bboxes=bboxes)
            else:
                print("No rectangle drawn. Please draw a rectangle.")
                return

        elif self.prompt_mode == 2:  # Text mode
            results = self.inferencer.infer(self.img_path, text_prompt=self.text_prompt)

        # Display results
        print("Inference completed. Displaying results...")
        results[0].show()

        # Ask if user wants to save
        save_choice = input("Do you want to save the results? (y/n): ").lower()
        if save_choice == "y" or save_choice == "yes":
            filename = input("Enter filename to save (e.g., 'sam_results.jpg'): ")
            results[0].save(filename=filename)
            print(f"Results saved as {filename}")
        else:
            print("Results not saved")

    def save_results(self):
        """Save the current image with prompts."""
        display_img = self.draw_ui_elements()
        filename = input("Enter filename to save current image (e.g., 'sam_prompts.jpg'): ")
        cv2.imwrite(filename, display_img)
        print(f"Image saved as {filename}")


# Run the UI if this file is executed directly
if __name__ == "__main__":
    # Ask user for image path
    img_path = input("Enter image path (press Enter for default): ").strip()
    if not img_path:
        img_path = r"D:\Min\Projects\VSCodeProjects\ultralytics-main\train\test_imgs\img1.jpg"

    # Create and run the UI
    sam_ui = SAMUI(img_path)
    sam_ui.run()
