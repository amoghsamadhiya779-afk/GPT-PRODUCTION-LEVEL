@echo off
echo Starting Math Persona Training...
venv_prod\Scripts\python.exe training\finetune_instruct.py --data data\math_data.jsonl --adapter_name math_adapter --epochs 5

echo.
echo Starting Physics Persona Training...
venv_prod\Scripts\python.exe training\finetune_instruct.py --data data\physics_data.jsonl --adapter_name physics_adapter --epochs 5

echo.
echo Starting General Assistant Persona Training...
venv_prod\Scripts\python.exe training\finetune_instruct.py --data "data\general_data.jsonl,data\rag_data.jsonl" --adapter_name general_adapter --epochs 5

echo.
echo All Persona Trainings Completed Successfully!
