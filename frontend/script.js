document.getElementById("video-upload-form").addEventListener("submit", async function (e) {
    e.preventDefault();
  
    const videoInput = document.getElementById("video");
    if (!videoInput.files.length) {
      alert("Please select a video file.");
      return;
    }
  
    const videoFile = videoInput.files[0];
  
    // Create a FormData object to send the video file
    const formData = new FormData();
    formData.append("file", videoFile);
  
    try {
      // Show a loading message
      const responseContainer = document.getElementById("response-container");
      responseContainer.innerHTML = "Processing... Please wait.";
  
      // Call the backend API
      const response = await fetch("http://localhost:8000/analyze-video/", {
        method: "POST",
        body: formData,
      });
  
      if (!response.ok) {
        throw new Error(`Server error: ${response.statusText}`);
      }
  
      // Get the response data
      const result = await response.json();
  
      // Display the results
      responseContainer.innerHTML = `
        <h3>Analysis Results</h3>
        <p><strong>Transcription:</strong> ${result.transcription}</p>
        <p><strong>Language:</strong> ${result.language}</p>
        <p><strong>Human Face Detected:</strong> ${result.human_face_detected}</p>
        <p><strong>NSFW Content Detected:</strong> ${result.nsfw_detected}</p>
        <p><strong>Inappropriate Language Detected:</strong> ${result.inappropriate_language_detected}</p>
      `;
    } catch (error) {
      console.error("Error:", error);
      document.getElementById("response-container").innerHTML = `<p>Error: ${error.message}</p>`;
    }
  });
  