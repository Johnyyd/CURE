// Khi chọn file CSV, gửi file lên server để lấy danh sách trường số và hiển thị checkbox
document.getElementById('fileInput').addEventListener('change', async function () {
    const fileInput = this;
    const fieldSelectionDiv = document.getElementById('fieldSelection');
    fieldSelectionDiv.innerHTML = 'Đang tải trường...';

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    try {
        const response = await fetch('/get-fields', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.error) {
            fieldSelectionDiv.innerHTML = `Lỗi: ${data.error}`;
            return;
        }

        if (data.length === 0) {
            fieldSelectionDiv.innerHTML = 'Không tìm thấy trường số nào trong file.';
            return;
        }

        // Tạo checkbox cho các trường số
        const htmlCheckboxes = data.map(field => `
            <label>
                <input type="checkbox" name="fields" value="${field}" checked />
                ${field}
            </label><br>
        `).join('');

        fieldSelectionDiv.innerHTML = htmlCheckboxes;
    } catch (err) {
        fieldSelectionDiv.innerHTML = 'Lỗi khi tải trường: ' + err.message;
    }
});

// Xử lý submit form: gửi file + trường đã chọn lên /analyze, nhận dữ liệu và vẽ biểu đồ
document.getElementById('uploadForm').addEventListener('submit', async function (e) {
    e.preventDefault();

    const fileInput = document.getElementById('fileInput');
    const checkboxes = document.querySelectorAll('input[name="fields"]:checked');
    const selectedFields = Array.from(checkboxes).map(cb => cb.value);

    if (!fileInput.files.length) {
        alert('Vui lòng chọn file CSV trước.');
        return;
    }

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    formData.append('fields', JSON.stringify(selectedFields));

    try {
        const response = await fetch('/analyze', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.error) {
            alert('Lỗi server: ' + data.error);
            return;
        }

        const { pca, labels, groups, cluster_labels, original_cluster_labels } = data;

        const x = pca.map(d => d[labels[0]]);
        const y = pca.map(d => d[labels[1]]);
        const z = pca.map(d => d[labels[2]]);

        // Vẽ biểu đồ sau PCA
        Plotly.newPlot('plot', [{
            x: x,
            y: y,
            z: z,
            mode: 'markers',
            type: 'scatter3d',
            marker: {
                size: 6,
                color: cluster_labels,
                colorscale: 'Rainbow',
                opacity: 0.8
            },
            text: groups
        }], {
            title: 'Phân cụm sau PCA',
            scene: {
                xaxis: { title: labels[0] },
                yaxis: { title: labels[1] },
                zaxis: { title: labels[2] }
            }
        });

        // Vẽ biểu đồ trước PCA
        const fieldsToUse = selectedFields.slice(0, 3);
        const dfText = await fileInput.files[0].text();
        const parsedCSV = Papa.parse(dfText, { header: true }).data;

        const xOrig = parsedCSV.map(row => parseFloat(row[fieldsToUse[0]]));
        const yOrig = parsedCSV.map(row => parseFloat(row[fieldsToUse[1]]));
        const zOrig = parsedCSV.map(row => parseFloat(row[fieldsToUse[2]]));

        Plotly.newPlot('plotOriginal', [{
            x: xOrig,
            y: yOrig,
            z: zOrig,
            mode: 'markers',
            type: 'scatter3d',
            marker: {
                size: 6,
                color: original_cluster_labels,
                colorscale: 'Rainbow',
                opacity: 0.8
            },
            text: groups
        }], {
            title: 'Phân cụm trước PCA',
            scene: {
                xaxis: { title: fieldsToUse[0] },
                yaxis: { title: fieldsToUse[1] },
                zaxis: { title: fieldsToUse[2] }
            }
        });

    } catch (err) {
        alert('Lỗi khi phân tích: ' + err.message);
    }
});
