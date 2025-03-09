import { useState } from 'react';
import axios from 'axios';

const API_URL = "http://localhost:8000/predict/";

function App() {
  const [file, setFile] = useState(null);
  const [prediction, setPrediction] = useState(null);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      alert("Please select a file first.");
      return;
    }
  
    const formData = new FormData();
    formData.append("file", file);
  
    try {
      const { data } = await axios.post(API_URL, formData);
      console.log("API Response:", data); // Debugging log
      
      setPrediction(`Prediction: ${data.predicted_label}`);
    } catch (error) {
      console.error("Error:", error);
      setPrediction("Error making prediction.");
    }
  };

  return (
    <div>
      <h1>Skin Cancer Detector</h1>
      <form onSubmit={handleSubmit}>
        <input type="file" onChange={handleFileChange} />
        <button type="submit">Predict</button>
      </form>
      {prediction !== null && <p>Prediction: {prediction}</p>}
    </div>
  );
}

export default App;
