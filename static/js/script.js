$(document).ready(function() {
    $('#upload-form').on('submit', function(event) {
        event.preventDefault();
        var formData = new FormData();
        formData.append('file', $('#csv-file')[0].files[0]);

        $.ajax({
            url: '/LangChainAgent/upload',
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function() {
                window.location.href = '/dashboard';
            }
        });
    });

    $('#auto-upload-form').on('submit', function(event) {
        event.preventDefault();
        var formData = new FormData();
        formData.append('file', $('#auto-csv-file')[0].files[0]);

        $.ajax({
            url: '/LangChainAgent/upload',
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function() {
                window.location.href = '/dashboard';
            }
        });
    });

    // Select last uploaded file
    $('#load-last-uploaded').on('click', function() {
        var lastFile = $('#file-select option:last').val();
        if (lastFile) {
            $('#file-select').val(lastFile);
            $('#load-insights').click();
        }
    });

    if (window.location.pathname === '/dashboard') {
        $('#overlay').show(); // Show overlay
        $('#preloader').show(); // Show preloader
        $.ajax({
            url: '/file-insights',
            type: 'POST',
            contentType: 'application/json', // Set content type to application/json
            data: JSON.stringify({ filename: $('#file-select').val() }), // Send data as JSON
            success: function(data) {
                $('#overlay').hide(); // Hide overlay
                $('#preloader').hide(); // Hide preloader
                displayCSVPreview(data.columns, data.rows);
                displayFileInsights(data.insights);
            },
            error: function() {
                $('#overlay').hide(); // Hide overlay on error
                $('#preloader').hide(); // Hide preloader on error
            }
        });
    }

    $('#toggle-sidebar').on('click', function() {
        $('.sidebar').toggleClass('hidden');
    });

    $('#load-insights').on('click', function() {
        var selectedFile = $('#file-select').val();
        if (selectedFile) {
            $('#overlay').show(); // Show overlay
            $('#preloader').show(); // Show preloader
            $.ajax({
                url: '/file-insights',
                type: 'POST',
                contentType: 'application/json',
                data: JSON.stringify({ filename: selectedFile }),
                success: function(data) {
                    $('#overlay').hide(); // Hide overlay
                    $('#preloader').hide(); // Hide preloader
                    displayCSVPreview(data.columns, data.rows);
                    displayFileInsights(data.insights);
                },
                error: function() {
                    $('#overlay').hide(); // Hide overlay on error
                    $('#preloader').hide(); // Hide preloader on error
                }
            });
        }
    });
});

function displayCSVPreview(columns, rows) {
    var thead = $('#csv-preview-header');
    var tbody = $('#csv-preview-body');
    
    thead.empty();
    tbody.empty();
    
    var headerRow = columns.map(col => `<th>${col}</th>`).join('');
    thead.append(headerRow);
    
    rows.forEach(row => {
        var rowHtml = '<tr>' + row.map(cell => `<td>${cell}</td>`).join('') + '</tr>';
        tbody.append(rowHtml);
    });
}

function displayFileInsights(insights) {
    // Update summary section
    const overview = insights.dataset_overview;
    $('#summary-rows').text(overview.total_rows.toLocaleString());
    $('#summary-columns').text(overview.total_columns.toLocaleString());
    $('#summary-missing').text(overview.missing_values_percentage + '%');
    $('#summary-duplicates').text(overview.duplicate_rows.toLocaleString());
    
    // Calculate total outliers percentage
    const totalOutliers = Object.values(insights.statistical_insights.outliers_per_column)
        .reduce((sum, count) => sum + count, 0);
    const outliersPercentage = ((totalOutliers / overview.total_rows) * 100).toFixed(1);
    $('#summary-outliers').text(outliersPercentage + '%');

    // Rest of the existing display logic
    var insightsCards = $('#insights-cards');
    insightsCards.empty();
    
    // Dataset Overview
    const quality = insights.data_quality;
    insightsCards.append(createCard('Dataset Overview', `
        <div class="kpi-grid summary-item">
            <div class="kpi-item value normal">
                <div class="kpi-number ">${overview.total_rows.toLocaleString()}</div>
                <div class="kpi-label">Total Rows</div>
            </div> 
            <div class="kpi-item value caution">
                <div class="kpi-number">${overview.total_columns.toLocaleString()}</div>
                <div class="kpi-label">Total Columns</div>
            </div>
            <div class="kpi-item value warning">
                <div class="kpi-number ${overview.missing_values_percentage > 20 ? 'warning' : ''}">${overview.missing_values_percentage}%</div>
                <div class="kpi-label">Missing Values</div>
                <div class="progress-bar">
                    <div class="fill" style="width: ${Math.min(overview.missing_values_percentage, 100)}%"></div>
                </div>
            </div>
            <div class="kpi-item value success">
                <div class="caution kpi-number ${overview.duplicate_rows > (overview.total_rows * 0.05) ? 'warning' : ''}">${overview.duplicate_rows.toLocaleString()}</div>
                <div class="kpi-label caution">Duplicate Rows</div>
            </div>
        </div>
    `));
    
    // Missing Values Analysis
    const missingValuesHtml = quality.columns_with_high_missing_values.map(col => {
        const missingData = quality.missing_values_by_column[col];
        return `<div class="missing-value-item">
            <div class="col-name caution">${col}</div>
            <div class="missing-stats">
                <span class="missing-percent ${missingData.percentage > 50 ? 'warning' : ''}">${missingData.percentage}%</span>
                <div class="progress-bar">
                    <div class="fill" style="width: ${missingData.percentage}%"></div>
                </div>
            </div>
        </div>`;
    }).join('');

    insightsCards.append(createCard('Columns with High Missing Values', `
        <div class="missing-values-list">
            ${missingValuesHtml || '<p>No columns with high missing values</p>'}
        </div>
    `));
    
    // Data Quality
    insightsCards.append(createCard('Data Quality', `
        <strong>Most Frequent Type:</strong> ${quality.most_frequent_data_type}<br>
        <strong>High Missing Columns:</strong>
        <ul>${quality.columns_with_high_missing_values.map(col => `<li>${col}</li>`).join('')}</ul>
    `));
    
    // Statistical Insights
    const stats = insights.statistical_insights;
    const outliersList = stats.outliers_per_column ? 
        Object.entries(stats.outliers_per_column)
            .filter(([_, count]) => count > 0)
            .map(([col, count]) => `<li>${col}: ${count} outliers</li>`)
            .join('') : '';
            
    insightsCards.append(createCard('Statistical Insights', `
        <strong>Skewed Columns:</strong> <span class="${stats.skewed_columns_count > 3 ? 'warning' : ''}">${stats.skewed_columns_count}</span><br>
        <strong>Strong Correlations:</strong> ${stats.correlation_pairs_count || 0} pairs<br>
        ${outliersList ? `<strong>Outliers Summary:</strong><ul>${outliersList}</ul>` : ''}
    `));
}

function createCard(title, content) {
    return `<div class="card">
        <h5>${title}</h5>
        <div class="content">${content}</div>
    </div>`;
}

function formatInsightContent(insight) {
    if (typeof insight === 'object') {
        return `<ul>${Object.entries(insight).map(([subKey, value]) => `<li><strong>${subKey}:</strong> ${value}</li>`).join('')}</ul>`;
    } else {
        return insight;
    }
}