$(document).ready(function() {
   

    $('#auto-upload-form').on('submit', function(event) {
    event.preventDefault();
    var formData = new FormData();
    var file = $('#auto-csv-file')[0].files[0];
    if (!file) return;

    formData.append('file', file);
    $('#upload-progress').show();

    $.ajax({
        url: '/LangChainAgent/upload',
        type: 'POST',
        data: formData,
        processData: false,
        contentType: false,
        xhr: function() {
            var xhr = new window.XMLHttpRequest();
            xhr.upload.addEventListener("progress", function(evt) {
                if (evt.lengthComputable) {
                    var percentComplete = (evt.loaded / evt.total) * 100;
                    $('#progress-bar').css('width', percentComplete + '%');
                }
            }, false);
            return xhr;
        },
        success: function() {
            window.location.href = 'LangChainAgent/dashboard';
        },
        error: function(xhr, status, error) {
            alert('Upload failed: ' + error);
        }
    });
});

});