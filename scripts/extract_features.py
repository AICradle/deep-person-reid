import json
import logging
import os
import numpy as np

from argparse import ArgumentParser

from torchreid.utils import FeatureExtractor
from torchreid.utils.tools import read_jsonl

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)


def main(in_fp, model_fp, out_dir):
    out_fp = os.path.join(out_dir, "features.")

    extractor = FeatureExtractor(
        model_name='osnet_x1_0',
        model_path=model_fp,
        device='cuda'
    )

    manifest_entries = read_jsonl(in_fp)
    with open(out_fp, "w") as f:
        for manifest_entry in manifest_entries:
            path = manifest_entry["path"]
            features = extractor([path])
            f.write(json.dumps(
                dict(path=path, features=features[0]), cls=NumpyEncoder)
            )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--manifest_fp", required=True, type=str)
    parser.add_argument("--model_fp", required=True, type=str)
    parser.add_argument("--out_dir", required=True, type=str)

    args = parser.parse_args()
    main(
        in_fp=args.manifest_fp,
        model_fp=args.model_fp,
        out_dir=args.out_dir
    )
