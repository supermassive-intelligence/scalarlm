# Diff between rschiavi/vllm-adapter and greg.vllm-adapter for vllm_model_manager.py

"""

(these changes I made when having issues with model not being found/LoRA adapter)

GREG'S CHANGES:

REMOVED DYNAMIC MODEL DISCOVERY (CRITICAL DIFFERENCE):

Your version has this critical functionality in find_model():

    # DYNAMIC DISCOVERY: Check if model exists in training directory
    training_dir = config.get("training_job_directory", "/app/cray/jobs")
    model_path = os.path.join(training_dir, model_name)
    
    if os.path.exists(model_path):
        # Verify it has model files (like .pt files) 
        pt_files = list(Path(model_path).glob("*.pt"))
        if len(pt_files) > 0:
            # Auto-register the discovered model
            self.register_model(model_name)
            return model_name

IMPACT:

"""

# ACTUAL GIT DIFF:
"""
--- infra/cray_infra/training/vllm_model_manager.py	2025-08-26 10:31:11
+++ /Users/rich/projects/scalarlm-greg/infra/cray_infra/training/vllm_model_manager.py	2025-08-29 10:07:55
@@ -27,31 +27,14 @@
         self._models[start_time] = model
 
     def find_model(self, model_name):
-        import os
-        from pathlib import Path
-        
         config = get_config()
 
-        # Check if it's the base model
         if model_name == config["model"]:
             return model_name
 
-        # Check if it's in registered models
         if model_name in set(self._models.values()):
             return model_name
 
-        # DYNAMIC DISCOVERY: Check if model exists in training directory
-        training_dir = config.get("training_job_directory", "/app/cray/jobs")
-        model_path = os.path.join(training_dir, model_name)
-        
-        if os.path.exists(model_path):
-            # Verify it has model files (like .pt files) 
-            pt_files = list(Path(model_path).glob("*.pt"))
-            if len(pt_files) > 0:
-                # Auto-register the discovered model
-                self.register_model(model_name)
-                return model_name
-
         return None
"""
