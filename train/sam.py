from ultralytics import SAM


class SAMInferencer:
    def __init__(self, model_path=r"D:\Min\Projects\VSCodeProjects\ultralytics-main\weights\sam2.1_t.pt"):
        # Load the SAM model
        self.model = SAM(model_path)
        # Display model information
        self.model.info()

    def infer(self, img_path, points=None, labels=None, bboxes=None, text_prompt=None):
        """Run inference with different types of prompts.

        Args:
            img_path: Path to the input image
            points: List of points [[x1, y1], [x2, y2], ...]
            labels: List of labels [1, 0, ...] where 1 is positive and 0 is negative
            bboxes: List of bounding boxes [x1, y1, x2, y2]
            text_prompt: Text prompt for segmentation

        Returns:
            Results from SAM model inference
        """
        # Run inference based on the provided prompts
        if points is not None and labels is not None:
            results = self.model(img_path, points=points, labels=labels)
        elif bboxes is not None:
            results = self.model(img_path, bboxes=bboxes)
        elif text_prompt is not None:
            # Note: SAM doesn't natively support text prompts, this is for future extension
            results = self.model(img_path)
        else:
            # If no prompt is provided, run without any prompt
            results = self.model(img_path)

        return results


# Example usage (if run directly)
if __name__ == "__main__":
    inferencer = SAMInferencer()
    img_path = r"D:\Min\Projects\VSCodeProjects\ultralytics-main\train\test_imgs\img1.jpg"

    # Example: Inference with points
    results = inferencer.infer(img_path, points=[[900, 370]], labels=[1])
    results[0].show()
