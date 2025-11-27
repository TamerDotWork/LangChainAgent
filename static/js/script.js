document.getElementById('uploadForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const fileInput = document.getElementById('fileInput');
            const errorDiv = document.getElementById('errorMsg');
            const loader = document.getElementById('loading-overlay');

            if (!fileInput.files[0]) {
                alert("Please select a file");
                return;
            }

            loader.style.display = 'block';
            errorDiv.classList.add('d-none');

            const formData = new FormData();
            formData.append('file', fileInput.files[0]);

            try {
                // FIX: Use 'upload' (relative) instead of '/upload' (absolute)
                const response = await fetch('upload', {
                    method: 'POST',
                    body: formData
                });

                // Read text first to catch HTML errors (like 404/500)
                const responseText = await response.text();

                let result;
                try {
                    result = JSON.parse(responseText);
                } catch (err) {
                    console.error("Server HTML Response:", responseText);
                    throw new Error(`Server returned error (${response.status}). See console.`);
                }

                if (!response.ok || result.error) {
                    throw new Error(result.error || "Unknown Server Error");
                }

                // Success: Save and Redirect
                sessionStorage.setItem('dqResult', JSON.stringify(result));
                window.location.href = "dashboard"; // Relative redirect

            } catch (err) {
                loader.style.display = 'none';
                errorDiv.textContent = err.message;
                errorDiv.classList.remove('d-none');
            }
        });

        function openFileOption() {
            document.getElementById("ajaxFileInput").click();
        }

        // Listen for when a file is selected
        document.getElementById("ajaxFileInput").addEventListener("change", function() {
            // Check if a file was actually selected
            if (this.value) {
                // Submit the form automatically
                document.getElementById("uploadForm").submit();
            }
        });
