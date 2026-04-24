from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["UI Tools"])

ROI_HTML = """
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ROI Drawing Tool</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-body: #0f172a;
            --bg-card: #1e293b;
            --primary: #3b82f6;
            --primary-hover: #2563eb;
            --text-main: #f8fafc;
            --text-sub: #cbd5e1;
            --border: #334155;
            --canvas-bg: #000000;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', sans-serif;
        }

        body {
            background-color: var(--bg-body);
            color: var(--text-main);
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            padding: 2rem;
        }

        .container {
            background-color: var(--bg-card);
            border-radius: 16px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.5);
            padding: 2rem;
            width: 100%;
            max-width: 1200px;
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }

        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .header h1 {
            font-size: 1.50rem;
            font-weight: 600;
            color: #fff;
        }

        .controls {
            display: flex;
            gap: 1rem;
            align-items: center;
            flex-wrap: wrap;
        }

        .btn {
            background-color: var(--primary);
            color: white;
            border: none;
            padding: 0.6rem 1.2rem;
            border-radius: 8px;
            font-size: 0.9rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s ease;
            box-shadow: 0 4px 6px rgba(59, 130, 246, 0.3);
        }

        .btn:hover {
            background-color: var(--primary-hover);
            transform: translateY(-1px);
        }

        .btn-danger {
            background-color: #ef4444;
            box-shadow: 0 4px 6px rgba(239, 68, 68, 0.3);
        }
        
        .btn-danger:hover {
            background-color: #dc2626;
        }

        .workspace {
            display: flex;
            gap: 1.5rem;
            align-items: flex-start;
        }

        .canvas-container {
            flex: 1;
            background-color: var(--canvas-bg);
            border: 2px dashed var(--border);
            border-radius: 12px;
            overflow: hidden;
            position: relative;
            min-height: 400px;
            display: flex;
            justify-content: center;
            align-items: center;
        }

        canvas {
            max-width: 100%;
            cursor: crosshair;
        }
        
        .placeholder-text {
            color: var(--text-sub);
            position: absolute;
            pointer-events: none;
        }

        .sidebar {
            width: 350px;
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }

        .result-box {
            background-color: rgba(0, 0, 0, 0.2);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1rem;
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        .result-box label {
            font-size: 0.85rem;
            color: var(--text-sub);
            font-weight: 500;
        }

        textarea {
            width: 100%;
            height: 150px;
            background-color: #0f172a;
            color: #10b981;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 0.75rem;
            font-family: monospace;
            font-size: 0.85rem;
            resize: none;
        }
        
        textarea:focus {
            outline: none;
            border-color: var(--primary);
        }

        input[type="file"] {
            display: none;
        }

        .instructions {
            font-size: 0.85rem;
            color: var(--text-sub);
            line-height: 1.5;
        }
    </style>
</head>
<body>

<div class="container">
    <div class="header">
        <h1>Cấu hình Vùng Nhận Diện (ROI)</h1>
        <div class="controls">
            <label for="imageLoader" class="btn">Tải ảnh lên</label>
            <input type="file" id="imageLoader" accept="image/*">
            <button class="btn btn-danger" id="btnClear">Xóa vùng vẽ</button>
        </div>
    </div>

    <div class="workspace">
        <div class="canvas-container" id="canvasWrapper">
            <p class="placeholder-text" id="placeholder">Chưa có ảnh. Vui lòng tải một ảnh lên.</p>
            <canvas id="roiCanvas"></canvas>
        </div>

        <div class="sidebar">
            <div class="result-box">
                <label>Tọa độ Normalize JSON (chuẩn hóa 0-1)</label>
                <textarea id="jsonOutput" readonly></textarea>
                <button class="btn" id="btnCopy" style="width: 100%;">Sao chép chuỗi JSON</button>
            </div>
            
            <div class="instructions">
                <strong>Hướng dẫn:</strong>
                <ul style="margin-left: 1rem; margin-top: 0.5rem;">
                    <li>Click chuột trái lên ảnh để tạo các điểm đa giác.</li>
                    <li>Để hoàn thành đa giác, click gần điểm bắt đầu (điểm đầu tiên sẽ đổi màu).</li>
                    <li>Sau khi hoàn thành, copy chuỗi JSON và paste vào hệ thống hoặc Swagger UI.</li>
                </ul>
            </div>
        </div>
    </div>
</div>

<script>
    const canvas = document.getElementById('roiCanvas');
    const ctx = canvas.getContext('2d');
    const imageLoader = document.getElementById('imageLoader');
    const btnClear = document.getElementById('btnClear');
    const jsonOutput = document.getElementById('jsonOutput');
    const placeholder = document.getElementById('placeholder');
    const wrapper = document.getElementById('canvasWrapper');
    const btnCopy = document.getElementById('btnCopy');

    let img = new Image();
    let points = [];
    let isClosed = false;
    let actualImgWidth = 0;
    let actualImgHeight = 0;
    
    // Resize Variables (hiển thị canvas vừa với màn hình nhưng giữ tỷ lệ gốc)
    let displayScale = 1;

    imageLoader.addEventListener('change', handleImage, false);
    btnClear.addEventListener('click', resetDrawing);
    btnCopy.addEventListener('click', () => {
        jsonOutput.select();
        document.execCommand('copy');
        const oldText = btnCopy.innerText;
        btnCopy.innerText = "Đã copy!";
        setTimeout(() => btnCopy.innerText = oldText, 2000);
    });

    canvas.addEventListener('click', addPoint);
    canvas.addEventListener('mousemove', drawMouseMove);

    function handleImage(e) {
        if (!e.target.files[0]) return;
        const reader = new FileReader();
        reader.onload = function(event) {
            img.onload = function() {
                placeholder.style.display = 'none';
                actualImgWidth = img.width;
                actualImgHeight = img.height;
                resetDrawing();
                resizeCanvas();
            }
            img.src = event.target.result;
        }
        reader.readAsDataURL(e.target.files[0]);
    }

    function resizeCanvas() {
        if (!img.src) return;
        
        let containerWidth = wrapper.clientWidth;
        // let containerHeight = wrapper.clientHeight; // có thể để height tự động theo proportion
        
        displayScale = Math.min(containerWidth / actualImgWidth, 1); // không phóng to hơn ảnh gốc
        
        canvas.width = actualImgWidth * displayScale;
        canvas.height = actualImgHeight * displayScale;
        
        drawCanvas();
    }

    window.addEventListener('resize', resizeCanvas);

    function getMousePos(evt) {
        const rect = canvas.getBoundingClientRect();
        return {
            x: evt.clientX - rect.left,
            y: evt.clientY - rect.top
        };
    }

    function addPoint(e) {
        if (!img.src || isClosed) return;
        
        const pos = getMousePos(e);
        
        // Kiểm tra close polygon (click gần điểm đầu)
        if (points.length >= 3) {
            const firstPt = points[0];
            const dx = pos.x - firstPt.x;
            const dy = pos.y - firstPt.y;
            const dist = Math.sqrt(dx*dx + dy*dy);
            if (dist < 10) {
                isClosed = true;
                updateOutput();
                drawCanvas();
                return;
            }
        }
        
        points.push(pos);
        drawCanvas();
    }

    function drawMouseMove(e) {
        if (!img.src || isClosed || points.length === 0) return;
        const pos = getMousePos(e);
        drawCanvas(pos);
    }

    function resetDrawing() {
        points = [];
        isClosed = false;
        jsonOutput.value = "";
        drawCanvas();
    }

    function drawCanvas(mousePos = null) {
        if (!img.src) return;
        
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

        if (points.length === 0) return;

        // Vẽ các line
        ctx.beginPath();
        ctx.moveTo(points[0].x, points[0].y);
        for (let i = 1; i < points.length; i++) {
            ctx.lineTo(points[i].x, points[i].y);
        }
        if (isClosed) {
            ctx.lineTo(points[0].x, points[0].y);
            ctx.fillStyle = 'rgba(59, 130, 246, 0.3)';
            ctx.fill();
        } else if (mousePos) {
            ctx.lineTo(mousePos.x, mousePos.y);
        }
        
        ctx.strokeStyle = '#3b82f6';
        ctx.lineWidth = 2;
        ctx.stroke();

        // Vẽ các đỉnh
        for (let i = 0; i < points.length; i++) {
            ctx.beginPath();
            ctx.arc(points[i].x, points[i].y, 5, 0, 2 * Math.PI);
            ctx.fillStyle = (i === 0 && points.length >= 3 && !isClosed) ? '#10b981' : '#ffffff';
            ctx.fill();
            ctx.stroke();
        }
    }

    function updateOutput() {
        if (!isClosed) return;
        
        // Cần map point từ toạ độ hiển thị về tọa độ gốc sau đó chia width/height => normalize
        const normalizedPoints = points.map(p => {
            const actX = p.x / displayScale;
            const actY = p.y / displayScale;
            return [
                parseFloat((actX / actualImgWidth).toFixed(4)),
                parseFloat((actY / actualImgHeight).toFixed(4))
            ];
        });

        jsonOutput.value = JSON.stringify(normalizedPoints);
    }
</script>
</body>
</html>
"""

@router.get("/roi_tool", response_class=HTMLResponse, summary="Công cụ lấy tọa độ ROI")
async def get_roi_tool():
    """
    Trả về trang HTML công cụ cho phép tải ảnh lên và vẽ đa giác xuất ra JSON ROI.
    """
    return HTMLResponse(content=ROI_HTML)
