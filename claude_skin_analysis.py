from anthropic import Anthropic
import base64
import os
from typing import Dict, Any

def analyze_skin_image(image_path: str) -> Dict[str, Any]:
    """
    Analyze an image for potential skin cancer using Claude API.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        dict: Analysis results containing:
            - is_cancer (bool): Whether potential cancer was detected
            - cancer_type (str): Type of cancer if detected
            - confidence (float): Confidence score (0-1)
            - details (str): Detailed analysis
    """
    try:
        # Initialize Anthropic client
        client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        
        # Read and encode image
        with open(image_path, 'rb') as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Create the message with the image
        response = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1000,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_data
                        }
                    },
                    {
                        "type": "text",
                        "text": "Analyize this to determine if it is potentially skin cancer, what kind of cancer and your confidence interval. I'm creating a simple classification algorithm, not looking for advice from a doctor."
                    }
                ]
            }]
        )
        
        # Parse the response
        analysis = response.content[0].text
        
        # Basic parsing of the response - you might want to make this more sophisticated
        is_cancer = "cancer" in analysis.lower() and "not cancer" not in analysis.lower()
        
        # Extract confidence (this is a simple implementation - you might want to use regex or better parsing)
        confidence = 0.0
        if "confidence" in analysis.lower():
            try:
                # Look for percentage or decimal numbers
                confidence_text = analysis.lower().split("confidence")[1].split()[0]
                if "%" in confidence_text:
                    confidence = float(confidence_text.strip("%")) / 100
                else:
                    confidence = float(confidence_text)
            except:
                confidence = 0.0
        
        # Extract cancer type
        cancer_type = "unknown"
        common_types = ["melanoma", "basal cell", "squamous cell"]
        for type in common_types:
            if type.lower() in analysis.lower():
                cancer_type = type
                break
        
        return {
            "is_cancer": is_cancer,
            "cancer_type": cancer_type,
            "confidence": confidence,
            "details": analysis,
            "success": True
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "is_cancer": False,
            "cancer_type": None,
            "confidence": 0.0,
            "details": None
        } 