
import xml.sax.saxutils as saxutils

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
    def __init__(self , treeId):
        self.root = TreeNode("ROOT")
        self.routes = []
        self.routes_ids = []
        self.treeId = treeId

    def getRoot(self):
        return self.routes[0]
        
    def getId(self):
        return self.treeId
    
    def insert_route(self, route, route_id,logo_data=None):
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
        self.routes_ids.append(route_id)

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
        return all_routes , self.routes_ids

    def __repr__(self):
        return f"Tree(root={self.root})"


def save_tree_to_file(file_path, tree, keywords, logo) -> bool:
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f'<Tree id = "{tree.getId()}" root_element = "{tree.getRoot()}">')
            f.write('\n\t<Routes>')

            urls , ids = tree.get_all_routes()
            for url_index in range(len(urls)):
                try:
                    safe_url = saxutils.escape(urls[url_index])
                    safe_logo = saxutils.escape(logo if logo else "")
                    safe_keywords = [saxutils.escape(k) for k in (keywords or [])]

                    f.write(f'\n\t\t<Route id = "{ids[url_index]}">')
                    f.write(f'\n\t\t\t<Domain>{safe_url}</Domain>')
                    f.write(f'\n\t\t\t<Logo>{safe_logo}</Logo>')
                    f.write(f'\n\t\t\t<Keywords>')

                    for kw in safe_keywords:
                        f.write(f'\n\t\t\t\t<Keyword>{kw}</Keyword>')

                    f.write(f'\n\t\t\t</Keywords>')
                    f.write(f'\n\t\t</Route>')

                except Exception as e:
                    print(f"[!] Skipped URL due to write error: {urls[url_index]} — {e}")
                    continue  # Prevent breaking the full file on one bad entry

            f.write('\n\t</Routes>')
            f.write(f'\n</Tree>')
            print(f"[✓] Tree saved to {file_path}")
        return True

    except Exception as e:
        print(f"[Error] Failed to save tree: {e}")
        return False

    