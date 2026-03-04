import re
from typing import Any, Optional

from openai.types.chat import ChatCompletionMessageParam
# from ..memory import Memory


class FormatPrompt:
    """
    A class to handle the formatting of prompts based on a template,
    with support for system prompts and variable extraction.

    - **Attributes**:
        - `template` (str): The template string containing placeholders.
        - `system_prompt` (Optional[str]): An optional system prompt to add to the dialog.
        - `variables` (List[str]): A list of variable names extracted from the template.
        - `formatted_string` (str): The formatted string derived from the template and provided variables.
        - `bound_objects` (Dict[str, Any]): Dictionary of objects bound to the prompt for use in expressions.
    """

    def __init__(
        self,
        template: str,
        format_prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        # memory: Optional[Memory] = None,
    ) -> None:
        """
        - **Description**:
            - Initializes the FormatPrompt with a template, an optional system prompt, and bound objects.

        - **Args**:
            - `template` (str): The string template with variable placeholders.
            - `system_prompt` (Optional[str], optional): An optional system prompt. Defaults to None.
            - `**bound_objects`: Named objects to bind to the prompt for use in expressions.
        """
        self.template = template
        self.format_prompt = format_prompt  # Store the format prompt
        self.system_prompt = system_prompt  # Store the system prompt
        self.variables = self._extract_variables()
        self.formatted_string = ""  # To store the formatted string
        # self.memory = memory  # Store memory

    def _extract_variables(self) -> list[str]:
        """
        - **Description**:
            - Extracts variable names from the template string using regular expressions.
            - Looks for simple variables in the format {variable_name}.

        - **Returns**:
            - `List[str]`: A list of variable names found within the template.
        """
        return re.findall(r"\{(\w+)\}", self.template)

    def _is_safe_expression(self, expr: str) -> bool:
        """
        - **Description**:
            - Checks if an expression is safe to evaluate.
            - Only allows attribute access and dictionary access.
            - Supports both single and double quotes for dictionary keys.
            - Supports numeric index access.

        - **Args**:
            - `expr` (str): The expression to check.

        - **Returns**:
            - `bool`: True if the expression is safe, False otherwise.
        """
        # Support both string keys and numeric indices
        safe_pattern = r'^(profile|status|context)(\.[a-zA-Z_][a-zA-Z0-9_]*|\[(?:[\'"][a-zA-Z0-9_][a-zA-Z0-9_]*[\'"]|\d+)\])*$'
        return bool(re.match(safe_pattern, expr))

    async def _eval_expr(self, expr: str, context: dict) -> Any:
        """
        - **Description**:
            - Evaluates expressions using eval with safety checks.
            - Supports expressions like profile.xxx, status.xxx, context.xxx
            - Also supports square bracket notation like profile.xxx["yyy"]
            - For profile and status, uses async get method from memory
            - Supports nested dictionary access with square brackets

        - **Args**:
            - `expr` (str): The expression to evaluate.
            - `context` (dict): The evaluation context.

        - **Returns**:
            - `Any`: The result of the expression evaluation.
        """
        # Create evaluation context
        eval_context = {
            # "profile": self.memory.status if self.memory else {},
            # "status": self.memory.status if self.memory else {},
            "profile": {},
            "status": {},
            "context": context or {},
        }

        # Add safety check for the expression
        if not self._is_safe_expression(expr):
            raise ValueError(f"Unsafe expression: {expr}")

        try:
            # Parse the expression to handle profile and status differently
            if expr.startswith("context."):
                key = expr.split(".", 1)[1]
                return eval_context["context"].get(key, "Don't know")
            elif expr.startswith(("profile.", "status.")):
                # Get the base value using async get method
                base_key = expr.split(".", 1)[1].split("[")[0]
                # base_value = (
                #     await self.memory.status.get(base_key) if self.memory else None
                # )
                base_value = None

                # If there's no square bracket notation, return the base value
                if "[" not in expr:
                    return base_value

                # Extract the square bracket expression
                bracket_expr = expr[expr.find("[") :]
                # Create a safe evaluation context with the base value
                safe_context = {"value": base_value}
                # Evaluate the bracket expression
                return eval(f"value{bracket_expr}", {"__builtins__": {}}, safe_context)
            else:
                # For other expressions, use regular eval
                return eval(expr, {"__builtins__": {}}, eval_context)
        except Exception as e:
            print(f"Error evaluating expression '{expr}': {str(e)}")
            return "Don't know"

    async def format(
        self,
        context: Optional[dict] = None,
        **kwargs,
    ) -> str:
        """
        - **Description**:
            - Formats the template string using the provided context and keyword arguments.
            - Supports simple variables {var} for direct substitution.
            - Supports complex expressions ${expression} in three formats:
              - ${profile.xxx}: Retrieves values from memory.profile
              - ${status.xxx}: Retrieves values from memory.status
              - ${context.xxx}: Retrieves values from the provided context dictionary

        - **Args**:
            - `context` (Optional[dict]): Dictionary containing context values.
            - `**kwargs`: Variable names and their corresponding values to format the template.

        - **Returns**:
            - `str`: The formatted string.

        - **Raises**:
            - `KeyError`: If a placeholder in the template does not have a corresponding key in kwargs.
            - `ValueError`: If an expression has an invalid format.
        """
        # Create a dictionary to hold all formatting variables
        format_vars = {}

        # First add explicitly provided kwargs, these take highest precedence
        for key, value in kwargs.items():
            if key in self.variables:
                format_vars[key] = value

        eval_context = context if context else {}

        # Handle complex expressions in the template using ${expression} syntax
        complex_pattern = r"\$\{([^}]+?)\}"
        result = self.template

        # First, protect all ${...} expressions with a temporary marker
        protected_expressions = {}
        counter = 0

        def protect_expression(match):
            nonlocal counter
            placeholder = f"__EXPR_{counter}__"
            protected_expressions[placeholder] = match.group(0)
            counter += 1
            return placeholder

        # Replace all ${...} expressions with placeholders
        result = re.sub(complex_pattern, protect_expression, result)

        # Format simple variables
        try:
            result = result.format(**format_vars)
        except KeyError as e:
            raise KeyError(f"Missing required variable in template: {e}")
        except Exception as e:
            raise ValueError(f"Error formatting template: {e}")

        # Now evaluate and replace the protected expressions
        for placeholder, original_expr in protected_expressions.items():
            expr = original_expr[2:-1].strip()  # Remove ${ and }
            try:
                eval_result = await self._eval_expr(expr, eval_context)
                result = result.replace(
                    placeholder, str(eval_result) if eval_result is not None else ""
                )
            except Exception as e:
                print(f"Error evaluating expression '{expr}': {str(e)}")
                result = result.replace(placeholder, original_expr)

        self.formatted_string = result
        return self.formatted_string

    def to_dialog(self) -> list[ChatCompletionMessageParam]:
        """
        - **Description**:
            - Converts the formatted prompt and optional system prompt into a dialog format suitable for chat systems.

        - **Returns**:
            - `List[Dict[str, str]]`: A list representing the dialog with roles and content.
        """
        dialog = []
        if self.system_prompt:
            dialog.append(
                {"role": "system", "content": self.system_prompt}
            )  # Add system prompt if it exists
        if self.format_prompt:
            dialog.append(
                {
                    "role": "user",
                    "content": self.formatted_string + "\n" + self.format_prompt,
                }
            )
        else:
            dialog.append(
                {"role": "user", "content": self.formatted_string}
            )  # Add user content
        return dialog

    def log(self) -> None:
        """
        - **Description**:
            - Logs the details of the FormatPrompt instance, including the template,
              system prompt, extracted variables, and formatted string.
        """
        print(f"Prompt Template: {self.template}")
        print(f"Format Prompt: {self.format_prompt}")
        print(f"System Prompt: {self.system_prompt}")  # Log the system prompt
        print(f"Variables: {self.variables}")
        print(f"Formatted String: {self.formatted_string}")  # Log the formatted string
