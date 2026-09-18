"""The prompts a phrase grounder answers confidently and wrongly, and where a cell can send them.

The router reads the prompt text and nothing else and loads no model, so every verdict is free. A
prompt the grounder cannot represent does not fail: it comes back as a high-scoring box on the
wrong object, which is why a cell can route such prompts to a vision-language model instead.
"""

from willy import PerceptionSpec, RuleBasedRouter, load_tree

router = RuleBasedRouter()
prompts = ("a red cube", "not the red cube", "the largest cube", "the cube to the left of the tray",
           "every cube", "greif den kaputten Wuerfel")
for prompt in prompts:
    print(f"{prompt!r:36} {router.route(prompt).describe()}")

# The route a prompt wants and the route a cell has are two facts. The base tree: the models
# block every cell inherits, which grounds every prompt with the phrase grounder.
tree = load_tree(None)
print(PerceptionSpec.from_config(tree.app_config.models).resolve())

# The same cell with the router on and a vision-language model behind it, given in memory. When
# that model will not load, `on_unavailable: refuse` rejects the pick rather than grounding the
# prompt with the model that would answer it wrongly.
routed = tree.with_values({"models.pipeline.zero_shot.backend": "vlm",
                           "models.pipeline.router.enabled": True})
print(PerceptionSpec.from_config(routed.app_config.models).resolve())
