function uploadImage() {
    const fileInput = document.getElementById('imageInput');
    const responseContainer = document.getElementById('responseContainer');

    const file = fileInput.files[0];
    if (!file) {
        alert('Please select an image.');
        return;
    }

    const formData = new FormData();
    formData.append('image', file);

    fetch('/predict', {
        method: 'POST',
        body: formData,
    })
    .then(response => response.json())
    .then(data => {
        responseContainer.innerHTML = `<p>Classification Result: ${data.result}</p>`;
    })
    .catch(error => {
        console.error('Error:', error);
        responseContainer.innerHTML = `<p>Error occurred. Please try again.</p>`;
    });
}

function updateDisplay() {

    // Print to console
    console.log('Updating display...');

    var inputElement = document.getElementById('imageInput');
    var file = inputElement.files[0]; // Get the selected file
    
    // Select the canvas and get its context
    var canvas = document.getElementById('imageCanvas');
    var ctx = canvas.getContext('2d');

    // Create a new Image object
    var img = new Image();

    // Set the source of the image as the selected file
    img.src = URL.createObjectURL(file);

    // Add event listener to wait for the image to load
    img.addEventListener('load', function() {
        // Draw the image on the canvas
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    });

}
