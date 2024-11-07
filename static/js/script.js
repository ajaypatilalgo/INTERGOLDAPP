document.querySelectorAll('.download-link').forEach(link => {
    link.addEventListener('click', function(event) {
        event.preventDefault();

        const url = this.href;
        const filename = url.substring(url.lastIndexOf('/') + 1);

        fetch(url,
            {mode:'cors'}
        )
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok.');
                }
                return response.blob();
            })
            .then(blob => {
                const downloadUrl = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.style.display = 'none';
                a.href = downloadUrl;
                a.download = filename;
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(downloadUrl);
                document.body.removeChild(a);
            })
            .catch(error => {
                console.error('There was a problem with the fetch operation:', error);
                alert('Could not download the file. Please check the console for details.');
            });
    });
});

