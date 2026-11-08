"""Dataset creation pairs captions with images, not with uploaded .txt files.

app.py imports the training UI at module level, so these tests exec
create_dataset from source instead of importing the app.
"""
import ast
import os
import pathlib
import shutil
import tempfile
import unittest


def _load_create_dataset():
    source_path = pathlib.Path(__file__).resolve().parents[1] / "app.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    namespace = {"os": os, "shutil": shutil}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "create_dataset":
            exec(compile(ast.Module([node], []), str(source_path), "exec"), namespace)
            break
    else:
        raise AssertionError("create_dataset was not found in app.py")

    resized = []

    def resize_image(image_path, output_path, size):
        resized.append(image_path)

    namespace["resize_image"] = resize_image
    namespace["resolve_path_without_quotes"] = lambda path: path
    namespace["resized"] = resized
    return namespace["create_dataset"], resized


class CreateDatasetCaptionTests(unittest.TestCase):
    def setUp(self):
        self.create_dataset, self.resized = _load_create_dataset()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = self.tmp.name

    def _touch(self, name, contents=""):
        path = os.path.join(self.root, name)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(contents)
        return path

    def _read(self, folder, name):
        with open(os.path.join(folder, name), encoding="utf-8") as handle:
            return handle.read()

    def test_txt_between_images_does_not_shift_later_captions(self):
        # Repro for images uploaded together with caption files:
        # [cat.png, cat.txt, dog.png] used to write dog's caption from the
        # third textbox (often None) instead of the second.
        files = [
            self._touch("cat.png", "img"),
            self._touch("cat.txt", "uploaded cat caption"),
            self._touch("dog.png", "img"),
        ]
        destination = os.path.join(self.root, "dataset")
        captions = ["edited cat caption", "dog caption", "unused third caption"]

        self.create_dataset(destination, 512, files, *captions)

        self.assertEqual(self._read(destination, "cat.txt"), "uploaded cat caption")
        self.assertEqual(self._read(destination, "dog.txt"), "dog caption")
        self.assertEqual(
            [os.path.basename(path) for path in self.resized],
            ["cat.png", "dog.png"],
        )

    def test_caption_file_before_its_image_is_kept(self):
        # Upload order is caption file, then dog, then cat. Textboxes follow
        # image order (dog, then cat), not the raw file list.
        files = [
            self._touch("cat.txt", "from file"),
            self._touch("dog.png", "img"),
            self._touch("cat.png", "img"),
        ]
        destination = os.path.join(self.root, "dataset")

        self.create_dataset(destination, 512, files, "dog from ui", "cat from ui")

        self.assertEqual(self._read(destination, "cat.txt"), "from file")
        self.assertEqual(self._read(destination, "dog.txt"), "dog from ui")

    def test_missing_caption_writes_empty_file_instead_of_raising(self):
        files = [self._touch("cat.png", "img"), self._touch("dog.png", "img")]
        destination = os.path.join(self.root, "dataset")

        self.create_dataset(destination, 512, files, None, "dog caption")

        self.assertEqual(self._read(destination, "cat.txt"), "")
        self.assertEqual(self._read(destination, "dog.txt"), "dog caption")


if __name__ == "__main__":
    unittest.main()
