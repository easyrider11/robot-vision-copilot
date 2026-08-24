SHELL   := /bin/bash
PY      := .venv/bin/python
UV      := uv
VENV    := .venv
PORT    ?= 8080

# demo knobs, all overridable:  make demo-libero INJECT=grasp_slip BACKEND=mock
BACKEND ?= auto
ENV     ?= auto
INJECT  ?= none
# MODE empty = auto (tabletop -> subgoal, libero -> e2e). Comment kept off the
# assignment line: make would fold the intervening spaces into the value.
MODE    ?=
EXPLAIN ?= full
STEPS   ?= 160
SUITE   ?= libero_spatial
TASKIDX ?= 0
PLANNER ?= rule
DETECTOR ?= color

.DEFAULT_GOAL := help

# ---------------------------------------------------------------- stage 0 --
.PHONY: audit
audit: $(VENV)               ## Stage 0: 环境审计（只读，不装任何东西）
	@$(PY) -m rvc.runners.audit

# ---------------------------------------------------------------- setup ----
$(VENV):
	@$(UV) venv --python 3.11 $(VENV)
	@$(UV) pip install --python $(PY) -e .

.PHONY: setup
setup: $(VENV)               ## 建立虚拟环境 + 安装 Stage 1 最小依赖 (numpy, pillow)
	@echo "✓ Stage 1 环境就绪：$(PY)"

.PHONY: setup-api
setup-api: $(VENV)           ## 追加 Stage 2 依赖 (fastapi, uvicorn, httpx)
	@$(UV) pip install --python $(PY) -e ".[api]"

.PHONY: setup-vla
setup-vla: $(VENV)           ## 追加真实 OpenVLA 依赖（需要 CUDA GPU，本机不适用）
	@echo "⚠ 这会安装 torch/transformers（数 GB），且只有在 CUDA GPU 上才有意义。"
	@echo "  权重另需 15.1 GB。先跑 'make audit' 确认。"
	@read -p "  继续? [y/N] " a; [[ $$a == y ]] && $(UV) pip install --python $(PY) -e ".[vla]"

.PHONY: setup-libero
setup-libero: $(VENV)        ## 安装并验证 LIBERO（约 2.6 GB，全部装在项目内）
	@bash scripts/setup_libero.sh

.PHONY: libero-tasks
libero-tasks: $(VENV)        ## 列出某个 LIBERO 任务套件里的所有任务
	@$(PY) -c "from rvc.envs.libero_env import list_tasks; \
	  [print(f'  {i:>2}. {t}') for i, t in list_tasks('$(SUITE)')]" 2>/dev/null \
	  || $(PY) -c "from rvc.envs.libero_env import probe; print('✗', probe()[1])"

# ---------------------------------------------------------------- stage 1 --
.PHONY: demo-libero
demo-libero: $(VENV)         ## Stage 1: 最小 OpenVLA/LIBERO 演示（自动降级到 mock）
	@$(PY) -m rvc.runners.demo_libero \
	  --backend $(BACKEND) --env $(ENV) --inject $(INJECT) \
	  --explain $(EXPLAIN) --max-steps $(STEPS) \
	  --libero-suite $(SUITE) --libero-task-index $(TASKIDX) \
	  --planner $(PLANNER) --detector $(DETECTOR) \
	  $(if $(MODE),--mode $(MODE),)

.PHONY: demo-recover
demo-recover: $(VENV)        ## Stage 1: 依次演示三种失败注入与 RECOVER 分支
	@for f in target_lost grasp_fail grasp_slip; do \
	  echo; echo "######## inject=$$f ########"; \
	  $(PY) -m rvc.runners.demo_libero --inject $$f --explain compact --max-steps 220; \
	done

.PHONY: demo-real
demo-real: $(VENV)           ## Stage 1: 拒绝降级——拿不到真实 VLA 就直接失败
	@$(PY) -m rvc.runners.demo_libero --backend $(BACKEND) --no-degraded --explain $(EXPLAIN)

# ---------------------------------------------------------------- stage 2 --
.PHONY: serve
serve: $(VENV)               ## Stage 2: 启动可观察服务 + Web 面板 (http://127.0.0.1:$(PORT))
	@$(PY) -m uvicorn rvc.service.app:app --reload --port $(PORT)

.PHONY: serve-vla
serve-vla: $(VENV)           ## 在 GPU 机器上启动远程 OpenVLA 推理服务
	@$(PY) -m uvicorn rvc.service.vla_server:app --host 0.0.0.0 --port 8000

.PHONY: smoke
smoke: $(VENV)               ## 冒烟测试：对已运行的服务打一遍所有接口
	@bash scripts/smoke_api.sh $(PORT)

# ---------------------------------------------------------------- stage 3 --
.PHONY: ros-up ros-build ros-shell
ros-up:                      ## Stage 3: 一键 colima + 构建镜像 + 无头冒烟（无 sudo）
	@bash scripts/setup_ros2.sh

ros-build:                   ## Stage 3: 构建 ROS 2 + Gazebo 镜像（需要 Docker）
	@command -v docker >/dev/null || { echo "✗ 未安装 Docker，见 docs/03-ros2-gazebo.md"; exit 1; }
	@docker compose -f ros2_ws/docker-compose.yml build

ros-shell:                   ## Stage 3: 进入 ROS 2 容器
	@docker compose -f ros2_ws/docker-compose.yml run --rm ros2 bash

.PHONY: ros-panda
ros-panda:                   ## Stage 3+: Panda 机械臂 + MoveIt Servo 的 pick-and-place
	@docker compose -f ros2_ws/docker-compose.yml run --rm ros2 \
	  ros2 launch rvc_agent panda.launch.py

# ---------------------------------------------------------------- dev ------
.PHONY: test lint clean clean-runs help
test: $(VENV)                ## 跑单元测试
	@$(UV) pip install --python $(PY) -q pytest && $(PY) -m pytest -q tests

lint: $(VENV)
	@$(UV) tool run ruff check src tests

clean-runs:                  ## 删除 runs/ 下的产物（只删本项目自己写的）
	@rm -rf runs/*/ && echo "✓ runs/ 已清空"

clean:
	@rm -rf $(VENV) .pytest_cache **/__pycache__ && echo "✓ 已清理"

help:
	@echo "Robot Vision Copilot — 分阶段目标"; echo
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'
	@echo; echo "常用变量: BACKEND=$(BACKEND) ENV=$(ENV) INJECT=$(INJECT) MODE=$(MODE) STEPS=$(STEPS) PLANNER=$(PLANNER) DETECTOR=$(DETECTOR)"

.PHONY: eval
eval: $(VENV)                ## 批量评测 Agent 栈：500 seeded 回合，输出恢复率/非法动作率/p95 时延
	@$(PY) -m rvc.runners.eval --episodes $(or $(EPISODES),500)

.PHONY: bc-data bc-train bc-eval
bc-data: $(VENV)             ## BC 基线 1/3：下载一个 LIBERO 任务的 50 条人类示教（~0.5 GB）
	@$(PY) -m rvc.runners.bc data --suite $(SUITE) --task-index $(TASKIDX)

bc-train: $(VENV)            ## BC 基线 2/3：本机训练 ResNet18x2+MLP 行为克隆（MPS，约 10 分钟）
	@$(PY) -m rvc.runners.bc train --suite $(SUITE) --task-index $(TASKIDX) --epochs $(or $(EPOCHS),30)

bc-eval: $(VENV)             ## BC 基线 3/3：在真实 LIBERO 上评测成功率（经 agent 运行时）
	@$(PY) -m rvc.runners.bc eval --suite $(SUITE) --task-index $(TASKIDX) --episodes $(or $(EPISODES),20)

.PHONY: play
smolvla-serve:               ## 本机起 SmolVLA-450M 推理服务（REAL VLA，.venv-lerobot，MPS）
	.venv-lerobot/bin/python -m rvc.service.smolvla_server --port 8100

smolvla-eval: $(VENV)        ## SmolVLA 在 LIBERO 官方初始状态上评测（先 make smolvla-serve）
	$(PY) -m rvc.runners.bc eval --policy smolvla --episodes 50 --max-steps 280

play: $(VENV)                ## 交互式 playground：自然语言指令 + 故障注入 + GIF 导出
	@$(PY) -m rvc.runners.play

.PHONY: yolo
yolo: $(VENV)                ## 合成数据 -> 微调 yolo11n (MPS) -> 评测 -> models/yolo-tabletop.pt
	@$(UV) pip install --python $(PY) -q -e ".[vision]" && $(PY) -m rvc.perception.yolo_train --epochs $(or $(EPOCHS),40)
