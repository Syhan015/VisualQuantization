# VisualQuantization

神经网络量化差异可视化平台。

## 项目简介

本项目旨在解决神经网络模型在量化（Quantization）压缩以实现边缘端高效部署时，产生的"结构变化黑盒化、精度损失不可见"的痛点。

构建一套基于 Web 的量化差异可视化平台，支持全精度（FP32）与量化版本（INT8）模型的同屏对比。

## 技术栈

### 前端
- React 18 + TypeScript
- Vite (构建工具)
- Zustand (状态管理)
- Netron (模型可视化)
- ECharts (图表渲染)
- Tailwind CSS

### 后端
- Python + FastAPI
- ONNX (模型解析)
- NumPy + SciPy (数值计算)

## 快速开始

### 前提条件

- Node.js >= 18
- Python >= 3.10

### Netron 部署

本软件包已集成 Netron v9 构建产物，无需额外下载。

系统依赖 Netron 进行 ONNX 模型可视化，需手动构建 web 版本：

1. 从 GitHub 下载 [Netron 源码](https://github.com/lutzroeder/netron)（以 `v9.0.5` 为例）
2. 在源码根目录执行 `node package.js build`
3. 将 `dist/web/` 目录下的所有文件复制到 `frontend/public/netron/`

### 前端

```bash
cd frontend
npm install
npm run dev
```

访问 http://localhost:5173

### 后端

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 9000
```

访问 http://localhost:9000/docs 查看 API 文档

## 项目结构

```
visual-quantization/
├── frontend/           # 前端 React 应用
│   ├── src/
│   │   ├── components/ # React 组件
│   │   ├── hooks/      # 自定义 Hooks
│   │   ├── services/   # API 服务
│   │   ├── stores/     # Zustand 状态
│   │   └── types/      # TypeScript 类型
│   └── public/
│       └── netron/     # Netron 源码 (需下载)
│
├── backend/            # 后端 FastAPI 应用
│   ├── app/
│   │   ├── api/        # API 路由
│   │   ├── core/       # 核心配置
│   │   ├── models/     # Pydantic 模型
│   │   └── services/   # 业务逻辑
│   └── tests/          # 测试
│
└── scripts/            # PyTorch 模型生成脚本
```

## 开发路线图

- [x] Phase 1: 基础框架搭建，单模型加载展示
- [x] Phase 2: 双栏联动，同步缩放平移
- [x] Phase 3: Diff 算法，结构差异提取
- [x] Phase 4: 节点高亮，ECharts 权重图表
- [x] Phase 5: 集成测试与优化

## License

MIT
