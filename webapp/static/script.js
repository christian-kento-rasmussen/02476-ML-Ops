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
