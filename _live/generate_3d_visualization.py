import json
import numpy as np
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=128
)

# Added a larger dataset to demonstrate higher dimensions
text = [
    "Tiger", "Lion", "Cheetah", "Leopard", 
    "Dog", "Cat", "Wolf", "Fox",
    "Gen AI Developer", "Data Scientist", "Machine Learning Engineer", "Software Engineer",
    "Doctor", "Nurse", "Surgeon", "Dentist",
    "Car", "Truck", "Motorcycle", "Bus",
    "Airplane", "Helicopter", "Drone", "Jet",
    "Apple", "Banana", "Orange", "Grape",
    "Laptop", "Smartphone", "Tablet", "Smartwatch"
]

print(f"Fetching embeddings for {len(text)} items...")
embedding_vector = embeddings.embed_documents(text)

# PCA using SVD
X = np.array(embedding_vector)
X_mean = np.mean(X, axis=0)
X_centered = X - X_mean
U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

# Compute up to 10 principal components
num_components = min(10, U.shape[1])
X_pca = U[:, :num_components] * S[:num_components]

# Normalize to a nice visual scale (-100 to 100 roughly)
max_val = np.max(np.abs(X_pca))
if max_val == 0: max_val = 1
X_pca_scaled = (X_pca / max_val) * 100

data = []
for i, t in enumerate(text):
    # Store all calculated components for this word
    components = [float(X_pca_scaled[i, c]) for c in range(num_components)]
    data.append({
        "label": t,
        "components": components
    })

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>High-Dimensional Vector Projector</title>
    <style>
        body {{
            margin: 0;
            overflow: hidden;
            background-color: #050510;
            font-family: 'Inter', 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            color: white;
        }}
        #info {{
            position: absolute;
            top: 20px;
            left: 20px;
            z-index: 100;
            pointer-events: none;
        }}
        h1 {{
            margin: 0;
            font-size: 24px;
            font-weight: 600;
            letter-spacing: 1px;
            background: -webkit-linear-gradient(45deg, #00f2fe, #4facfe);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        p {{
            margin-top: 5px;
            font-size: 14px;
            color: #8892b0;
        }}
        
        /* Glassmorphism Control Panel */
        #controls {{
            position: absolute;
            top: 20px;
            right: 20px;
            z-index: 100;
            background: rgba(10, 25, 47, 0.7);
            border: 1px solid rgba(100, 255, 218, 0.3);
            border-radius: 12px;
            padding: 20px;
            backdrop-filter: blur(10px);
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5);
            display: flex;
            flex-direction: column;
            gap: 15px;
            width: 250px;
        }}
        
        .control-group {{
            display: flex;
            flex-direction: column;
            gap: 5px;
        }}
        
        .control-group label {{
            font-size: 12px;
            color: #64ffda;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        select {{
            background: rgba(2, 12, 27, 0.7);
            color: white;
            border: 1px solid #233554;
            padding: 8px 12px;
            border-radius: 6px;
            font-family: 'Inter', sans-serif;
            font-size: 14px;
            outline: none;
            cursor: pointer;
            transition: border-color 0.3s;
        }}
        
        select:hover, select:focus {{
            border-color: #64ffda;
        }}

        .label {{
            color: #ffffff;
            font-family: 'Inter', sans-serif;
            font-size: 13px;
            font-weight: 500;
            padding: 6px 10px;
            background: rgba(10, 25, 47, 0.85);
            border: 1px solid rgba(100, 255, 218, 0.2);
            border-radius: 6px;
            backdrop-filter: blur(4px);
            pointer-events: auto;
            cursor: pointer;
            transition: all 0.3s ease;
        }}
        .label:hover {{
            background: rgba(100, 255, 218, 0.2);
            border-color: rgba(100, 255, 218, 0.8);
            transform: scale(1.1);
            z-index: 1000;
        }}
    </style>
</head>
<body>
    <div id="info">
        <h1>Vector Projector</h1>
        <p>Exploring {num_components} Dimensions</p>
    </div>
    
    <div id="controls">
        <div class="control-group">
            <label for="xAxis">X Axis</label>
            <select id="xAxis"></select>
        </div>
        <div class="control-group">
            <label for="yAxis">Y Axis</label>
            <select id="yAxis"></select>
        </div>
        <div class="control-group">
            <label for="zAxis">Z Axis</label>
            <select id="zAxis"></select>
        </div>
    </div>
    
    <script async src="https://unpkg.com/es-module-shims@1.8.0/dist/es-module-shims.js"></script>
    <script type="importmap">
        {{
            "imports": {{
                "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
                "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
            }}
        }}
    </script>

    <script type="module">
        import * as THREE from 'three';
        import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';
        import {{ CSS2DRenderer, CSS2DObject }} from 'three/addons/renderers/CSS2DRenderer.js';

        const data = {json.dumps(data)};
        const numComponents = {num_components};
        
        // Initial axes mapping
        let axisMap = {{ x: 0, y: 1, z: 2 }};

        // Populate dropdowns
        const selects = ['xAxis', 'yAxis', 'zAxis'];
        selects.forEach((id, index) => {{
            const select = document.getElementById(id);
            for(let i = 0; i < numComponents; i++) {{
                const option = document.createElement('option');
                option.value = i;
                option.text = `Principal Component ${{i+1}}`;
                if(i === index) option.selected = true;
                select.appendChild(option);
            }}
            
            select.addEventListener('change', (e) => {{
                const axis = id.charAt(0); // 'x', 'y', or 'z'
                axisMap[axis] = parseInt(e.target.value);
                updateTargetPositions();
            }});
        }});

        // Scene setup
        const scene = new THREE.Scene();
        scene.fog = new THREE.FogExp2(0x050510, 0.002);

        const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 2000);
        camera.position.set(200, 150, 250);

        const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        document.body.appendChild(renderer.domElement);

        const labelRenderer = new CSS2DRenderer();
        labelRenderer.setSize(window.innerWidth, window.innerHeight);
        labelRenderer.domElement.style.position = 'absolute';
        labelRenderer.domElement.style.top = '0px';
        labelRenderer.domElement.style.pointerEvents = 'none';
        document.body.appendChild(labelRenderer.domElement);

        const controls = new OrbitControls(camera, labelRenderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.autoRotate = true;
        controls.autoRotateSpeed = 0.5;

        // Lighting
        scene.add(new THREE.AmbientLight(0xffffff, 0.4));
        const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
        dirLight.position.set(100, 200, 50);
        scene.add(dirLight);

        // Grid
        const gridHelper = new THREE.GridHelper(400, 20, 0x00f2fe, 0x111122);
        gridHelper.position.y = -150;
        scene.add(gridHelper);

        // State arrays for animation
        const nodes = [];
        const currentPositions = [];
        const targetPositions = [];
        const nodeConnections = []; // to store line meshes

        // Node geometries
        const geometry = new THREE.SphereGeometry(3, 32, 32);
        const material = new THREE.MeshPhysicalMaterial({{
            color: 0x00f2fe, emissive: 0x00f2fe, emissiveIntensity: 0.5,
            metalness: 0.2, roughness: 0.1
        }});
        
        const lineMaterial = new THREE.LineBasicMaterial({{
            color: 0x4facfe, transparent: true, opacity: 0.2
        }});

        // Create nodes
        data.forEach((item, i) => {{
            const mesh = new THREE.Mesh(geometry, material);
            scene.add(mesh);
            nodes.push(mesh);
            
            // Initial positions based on default axes
            const x = item.components[axisMap.x];
            const y = item.components[axisMap.y];
            const z = item.components[axisMap.z];
            
            currentPositions.push(new THREE.Vector3(x, y, z));
            targetPositions.push(new THREE.Vector3(x, y, z));
            mesh.position.set(x, y, z);

            // Label
            const div = document.createElement('div');
            div.className = 'label';
            div.textContent = item.label;
            // Hide by default
            div.style.opacity = '0';
            div.style.pointerEvents = 'none';

            const label = new CSS2DObject(div);
            label.position.set(0, 8, 0);
            mesh.add(label);

            mesh.userData.labelDiv = div;
        }});

        // Set up connection lines (we'll update their geometry in the loop)
        for(let i=0; i<data.length; i++) {{
            // For each node, we create 2 lines for nearest neighbors
            const line1 = new THREE.Line(new THREE.BufferGeometry(), lineMaterial);
            const line2 = new THREE.Line(new THREE.BufferGeometry(), lineMaterial);
            scene.add(line1); scene.add(line2);
            nodeConnections.push({{ n1: line1, n2: line2 }});
        }}

        // Raycaster for click events
        const raycaster = new THREE.Raycaster();
        const mouse = new THREE.Vector2();

        window.addEventListener('click', (event) => {{
            // Ignore UI clicks
            if (event.target.tagName === 'SELECT' || event.target.tagName === 'LABEL') return;

            mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
            mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

            raycaster.setFromCamera(mouse, camera);
            const intersects = raycaster.intersectObjects(nodes);

            if (intersects.length > 0) {{
                const clickedMesh = intersects[0].object;
                const div = clickedMesh.userData.labelDiv;
                if (div) {{
                    if (div.style.opacity === '0') {{
                        div.style.opacity = '1';
                        div.style.pointerEvents = 'auto';
                    }} else {{
                        div.style.opacity = '0';
                        div.style.pointerEvents = 'none';
                    }}
                }}
            }}
        }});

        function updateTargetPositions() {{
            data.forEach((item, i) => {{
                targetPositions[i].set(
                    item.components[axisMap.x],
                    item.components[axisMap.y],
                    item.components[axisMap.z]
                );
            }});
        }}

        function animate() {{
            requestAnimationFrame(animate);
            
            // Smoothly interpolate (lerp) from current to target positions
            let positionsChanged = false;
            for(let i=0; i<data.length; i++) {{
                currentPositions[i].lerp(targetPositions[i], 0.05); // 0.05 is the transition speed
                nodes[i].position.copy(currentPositions[i]);
                
                // Add floating effect based on time
                const time = Date.now() * 0.001;
                nodes[i].position.y += Math.sin(time * 2 + i) * 0.1;
            }}

            // Update connections based on current visual positions
            for(let i=0; i<currentPositions.length; i++) {{
                let nearestDist1 = Infinity, nearestDist2 = Infinity;
                let nearest1 = -1, nearest2 = -1;
                
                for(let j=0; j<currentPositions.length; j++) {{
                    if(i === j) continue;
                    const dist = currentPositions[i].distanceTo(currentPositions[j]);
                    if(dist < nearestDist1) {{
                        nearestDist2 = nearestDist1;
                        nearest2 = nearest1;
                        nearestDist1 = dist;
                        nearest1 = j;
                    }} else if(dist < nearestDist2) {{
                        nearestDist2 = dist;
                        nearest2 = j;
                    }}
                }}
                
                if(nearest1 !== -1) {{
                    nodeConnections[i].n1.geometry.setFromPoints([currentPositions[i], currentPositions[nearest1]]);
                }}
                if(nearest2 !== -1) {{
                    nodeConnections[i].n2.geometry.setFromPoints([currentPositions[i], currentPositions[nearest2]]);
                }}
            }}

            controls.update();
            renderer.render(scene, camera);
            labelRenderer.render(scene, camera);
        }}

        window.addEventListener('resize', () => {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
            labelRenderer.setSize(window.innerWidth, window.innerHeight);
        }});

        animate();
    </script>
</body>
</html>
"""

with open("vector_visualization.html", "w") as f:
    f.write(html_content)

print("Generated vector_visualization.html successfully with dynamic dimensions!")
