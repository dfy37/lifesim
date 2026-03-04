class DotDict(dict):
    """
    A dictionary that supports dot notation access to its items.

    - **Description**:
        - Extends the standard dictionary to allow attribute-style access
        - Example: d = DotDict({'foo': 'bar'}); d.foo == 'bar'
        - Supports merging with other DotDict instances
        - Maintains reference to original dictionaries when merged

    - **Args**:
        - Same as dict constructor

    - **Returns**:
        - A dictionary with attribute-style access
    """

    def __getattr__(self, key):
        return self.get(key)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        if key in self:
            del self[key]

    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)
        # Convert nested dictionaries to DotDict
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = DotDict(value)
            elif isinstance(value, list):
                self[key] = [
                    DotDict(item) if isinstance(item, dict) else item for item in value
                ]

    def merge(self, other):
        """
        Merges another DotDict into this one.

        - **Description**:
            - Merges another DotDict into this one
            - Maintains references to original dictionaries
            - Updates are synchronized with original dictionaries

        - **Args**:
            - `other` (DotDict): The DotDict to merge with

        - **Returns**:
            - `self` (DotDict): The merged DotDict
        """
        if not isinstance(other, DotDict):
            other = DotDict(other)

        for key, value in other.items():
            if (
                key in self
                and isinstance(self[key], DotDict)
                and isinstance(value, DotDict)
            ):
                self[key].merge(value)
            else:
                self[key] = value
        return self

    def __or__(self, other):
        """
        Implements the | operator for merging DotDicts.

        - **Description**:
            - Allows using the | operator to merge DotDicts
            - Example: c = a | b

        - **Args**:
            - `other` (DotDict): The DotDict to merge with

        - **Returns**:
            - `DotDict`: A new merged DotDict
        """
        result = DotDict(self)
        return result.merge(other)

    def __ior__(self, other):
        """
        Implements the |= operator for in-place merging.

        - **Description**:
            - Allows using the |= operator for in-place merging
            - Example: a |= b

        - **Args**:
            - `other` (DotDict): The DotDict to merge with

        - **Returns**:
            - `self` (DotDict): The updated DotDict
        """
        return self.merge(other)