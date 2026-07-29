class DatasetEvaluator:
    def reset(self): pass
    def process(self, inputs, outputs): pass
    def evaluate(self): return {}

def inference_on_dataset(model, data_loader, evaluator): return {}

class _BaseEvaluator(DatasetEvaluator):
    def __init__(self, *args, **kwargs): pass

class CityscapesInstanceEvaluator(_BaseEvaluator): pass
class CityscapesSemSegEvaluator(_BaseEvaluator): pass
class COCOEvaluator(_BaseEvaluator): pass
class LVISEvaluator(_BaseEvaluator): pass

class DatasetEvaluators:
    def __init__(self, evaluators): self.evaluators = evaluators
    def reset(self): [e.reset() for e in self.evaluators]
    def process(self, inputs, outputs): [e.process(inputs, outputs) for e in self.evaluators]
    def evaluate(self): return {}

def verify_results(cfg, results): pass
