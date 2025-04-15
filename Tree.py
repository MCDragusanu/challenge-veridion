import os

class TreeNode:
    def __init__(self, value):
        self.value = value
        self.children = {}
        self.logos = []
        self.original_routes = []  # Stores original route strings

    def add_child(self, part):
        if part not in self.children:
            self.children[part] = TreeNode(part)
        return self.children[part]

    def __repr__(self):
        return f"TreeNode({self.value})"


class Tree:
    def __init__(self):
        self.root = TreeNode("ROOT")
        self.routes = []

    def insert_route(self, route, logo_data=None):
        """
        Inserts a full route like 'nike.com/shoes-men'.
        Splits on '.', '/', '-' for tree structure,
        but preserves original route string at leaf.
        """
        original_route = route.strip()
        parts = [p for p in route.replace('.', '/').replace('-', '/').strip("/").split("/") if p]

        current = self.root
        for part in parts:
            current = current.add_child(part)

        if logo_data:
            current.logos.append(logo_data)

        current.original_routes.append(original_route)
        self.routes.append(original_route)

    def get_all_routes(self):
        """
        Returns all original routes that were inserted.
        """
        all_routes = []

        def dfs(node):
            all_routes.extend(node.original_routes)
            for child in node.children.values():
                dfs(child)

        dfs(self.root)
        return all_routes

    def __repr__(self):
        return f"Tree(root={self.root})"


def save_tree_to_file(file_path, tree: Tree) -> bool:
    try:
        with open(file_path, 'w') as f:
            def dfs(node: TreeNode):
                for child_value, child_node in node.children.items():
                    f.write(f"{node.value} -> {child_value}\n")
                    dfs(child_node)
            dfs(tree.root)
        print(f"[✓] Tree saved to {file_path}")
        return True
    except Exception as e:
        print(f"[Error] Failed to save tree: {e}")
        return False


def load_tree_from_file(file_path) -> Tree:
    if not os.path.exists(file_path):
        print(f"Couldn't load tree structure from: {file_path}")
        return None

    tree = Tree()
    nodes = {"ROOT": tree.root}

    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if '->' not in line:
                    continue
                parent, child = map(str.strip, line.split('->', 1))

                if parent not in nodes:
                    nodes[parent] = TreeNode(parent)
                if child not in nodes:
                    nodes[child] = TreeNode(child)

                nodes[parent].children[child] = nodes[child]

        print(f"[✓] Tree loaded from {file_path}")
        return tree
    except Exception as e:
        print(f"[Error] Failed to load tree: {e}")
        return None
