import re
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from tkinter.scrolledtext import ScrolledText

# ============================
# Custom Exceptions
# ============================

class SemanticError(Exception):
    """Custom exception for semantic errors with line numbers."""
    pass

class CustomSyntaxError(Exception):
    """Custom exception for syntax errors with line numbers."""
    pass

# ============================
# Symbol Table Class
# ============================

class SymbolTable:
    def __init__(self, output_callback=None):
        # Stores variable information: {variable_name: {'type': type, 'initialized': bool, 'value': value}}
        self.variables = {}
        self.output_callback = output_callback

    def declare_variable(self, name, var_type='NOOB', line_number=None):
        if name in self.variables:
            raise SemanticError(f"Variable '{name}' already declared. Error at line {line_number}")
        self.variables[name] = {'type': var_type, 'initialized': False, 'value': None}
        self._log(f"Declared variable '{name}' of type '{var_type}'.")

    def initialize_variable(self, name, var_type, value, line_number=None):
        if name not in self.variables:
            raise SemanticError(f"Variable '{name}' not declared before initialization. Error at line {line_number}")
        self.variables[name].update({'type': var_type, 'initialized': True, 'value': value})
        self._log(f"Initialized variable '{name}' with type '{var_type}' and value '{value}'.")

    def is_compatible(self, current_type, var_type):
        """Check if var_type can be assigned to current_type with implicit casting."""
        compatibility = {
            'NOOB': True,
            'YARN': var_type in ['NUMBR', 'NUMBAR'],
            'NUMBR': var_type in ['NUMBR', 'NUMBAR', 'TROOF'],
            'NUMBAR': var_type in ['NUMBR', 'NUMBAR', 'TROOF'],
            'TROOF': var_type in ['TROOF', 'NUMBR', 'NUMBAR']
        }
        return compatibility.get(current_type, False) or current_type == var_type

    def assign_variable(self, name, var_type, value=None, line_number=None, is_casting=False):
        """Handles variable assignment and typecasting."""
        if name not in self.variables:
            raise SemanticError(f"Variable '{name}' not declared before assignment. Error at line {line_number}")
        
        current_type = self.variables[name]['type']
        
        # Handle explicit casting using 'IS NOW A' (when is_casting is True)
        if is_casting: 
            # If the current type and the new type are different, cast the value
            if current_type != var_type:
                value = self._cast_value(current_type, var_type, value)  # Cast the value
                self.variables[name].update({'type': var_type, 'value': value})
                self._log(f"Casted variable '{name}' to type '{var_type}' with value '{value}'.")
            return
        
        # Handle normal assignment (without casting)
        if not self.is_compatible(current_type, var_type) and current_type != var_type:
            raise SemanticError(f"Type mismatch: Cannot assign type '{var_type}' to variable '{name}' of type '{current_type}'. Error at line {line_number}")
        
        # Implicit casting when the types are compatible but different
        if self.is_compatible(current_type, var_type) and current_type != var_type:
            # Convert 'WIN' to 1 and 'FAIL' to 0 for numeric types
            if var_type in ['NUMBR', 'NUMBAR'] and value in ['WIN', 'FAIL']:
                value = 1 if value == 'WIN' else 0  
        
            print(value)
            
            value = self._cast_value(current_type, var_type, value)

                
        # Ensure that the variable's value is set correctly
        action = "Initialized" if current_type == 'NOOB' else "Assigned"
        
        # Update the variable with the correct value
        if var_type == 'TROOF' and value in ['WIN', 'FAIL']:
            if value == 'WIN':
                value = 'WIN'
            elif value == 'FAIL':
                print("AAAAAAAAAAAAA")
                value = 'FAIL'
        
        self.variables[name].update({'type': var_type, 'initialized': True, 'value': value})
        self._log(f"{action} variable '{name}' with type '{var_type}' and value '{value}'.")





    def _cast_value(self, current_type, var_type, value):
        """Handle implicit and explicit casting based on type compatibility."""
        
        # Casting to TROOF
        if var_type == 'TROOF':
            # Casting from NUMBR or NUMBAR
            if current_type in ['NUMBR', 'NUMBAR']:
                if value == 0:
                    return 'FAIL'
                elif value == 1 or value == 1.0:
                    return 'WIN'
                elif value == 'FAIL':
                    return 'FAIL'
                elif value == 'WIN':
                    return 'WIN'
                else:
                    return 'error'
                
            if value == 'WIN':
                return 'WIN'
            
            if value == 'FAIL':
                return 'FAIL'
            # Casting from YARN (empty string is FAIL, all other strings are WIN)
            if current_type == 'YARN':
                return 'FAIL' if value.strip() == '' else 'WIN'

            # Casting from NOOB (uninitialized) is always FAIL
            if current_type == 'NOOB':
                return 'FAIL'

            # All other values are treated as WIN (e.g., booleans or objects)
            return 'WIN' if value else 'FAIL'

        # Handle casting to NUMBR (integer)
        if var_type == 'NUMBR':
            if current_type == 'NUMBAR':
                return int(value)  # Convert NUMBAR (float) to NUMBR (int)
            if current_type == 'YARN':
                try:
                    return int(value)  # Try converting string to int
                except ValueError:
                    raise SemanticError(f"Cannot cast YARN '{value}' to NUMBR.")
        
        # Handle casting to NUMBAR (float)
        if var_type == 'NUMBAR':
            if current_type == 'NUMBR':
                return float(value)  # Convert NUMBR (int) to NUMBAR (float)
            if current_type == 'YARN':
                try:
                    return float(value)  # Try converting string to float
                except ValueError:
                    raise SemanticError(f"Cannot cast YARN '{value}' to NUMBAR.")
        
        # Handle casting to YARN (string)
        if var_type == 'YARN':
            return str(value)  # Convert any type to string
        
        # Handle casting to NOOB (uninitialized value)
        if var_type == 'NOOB':
            return None  # No value, represents uninitialized
        
        # Default case: return the original value if no casting is necessary
        return value



    def get_variable_type(self, name, line_number=None):
        if name not in self.variables:
            raise SemanticError(f"Variable '{name}' used before declaration. Error at line {line_number}")
        return self.variables[name]['type']

    def is_initialized(self, name, line_number=None):
        if name not in self.variables:
            raise SemanticError(f"Variable '{name}' used before declaration. Error at line {line_number}")
        var_type = self.variables[name]['type']
        if var_type == 'NOOB' or self.variables[name]['initialized']:
            return True
        raise SemanticError(f"Variable '{name}' used before initialization. Error at line {line_number}")

    def _log(self, message):
        """Helper method to log messages via the output callback."""
        if self.output_callback:
            self.output_callback(message)

    def __str__(self):
        """Return a string representation of the symbol table with only Identifier and Value."""
        lines = ["Symbol Table:"]
        for var, info in self.variables.items():
            value = info.get('value', 'None')
            if info['type'] == 'NOOB':
                value = 'NOOB'
            lines.append(f"  Identifier: {var}, Value: {value}")
        return "\n".join(lines)

# ============================
# Lexical Analyzer Class
# ============================

class LexicalAnalyzer:
    def __init__(self):
        # Define token regex patterns with comment tokens prioritized and refined
        token_specs = [
            # Program structure
            ('HAI', r'\bHAI\b'),                  # Start of program
            ('KTHXBYE', r'\bKTHXBYE\b'),

            # Comments (Case-Insensitive and Positioned)
            ('OBTW', r'\bOBTW\b'),                       # Start of multi-line comment
            ('TLDR', r'\bTLDR\b'),                       # End of multi-line comment
            ('single_line_comment', r'\bBTW\b[^\n]*'),  # Single-line comment

            # Flow-control and switch-case keywords
            ('O_RLY', r'\bO RLY\?'),                     # If statement start
            ('WTF', r'\bWTF\?'),                         # Start of switch-case

            # Typecasting
            ('MAEK', r'\bMAEK\b'),                       # Explicit typecasting operator
            ('IS_NOW_A', r'\bIS NOW A\b'),               # Re-casting operator
            ('TYPE', r'\b(NUMBAR|NUMBR|YARN|TROOF|NOOB)\b'),  # Data types

            # Boolean operations
            ('BOTH_OF', r'\bBOTH OF\b'),                 # AND operation
            ('EITHER_OF', r'\bEITHER OF\b'),             # OR operation
            ('WON_OF', r'\bWON OF\b'),                   # XOR operation
            ('NOT', r'\bNOT\b'),                         # NOT operation
            ('ALL_OF', r'\bALL OF\b'),                   # Infinite-arity AND
            ('ANY_OF', r'\bANY OF\b'),                   # Infinite-arity OR
            ('MKAY', r'\bMKAY\b'),                       # Closing infinite-arity operation

            # Comparison operations
            ('BOTH_SAEM', r'\bBOTH SAEM\b'),             # Equality
            ('DIFFRINT', r'\bDIFFRINT\b'),               # Inequality

            # Relational comparisons
            ('BIGGR_OF', r'\bBIGGR OF\b'),               # Maximum (>= or >)
            ('SMALLR_OF', r'\bSMALLR OF\b'),             # Minimum (<= or <)

            # Data types and literals
            ('NOOB', r'\bNOOB\b'),                       # Uninitialized variable type
            ('NUMBAR', r'-?\d+\.\d+'),                   # Floating point literal
            ('NUMBR', r'-?\d+'),                         # Integer literal
            ('YARN', r'"[^"]*"'),                        # String literal
            ('TROOF', r'\b(WIN|FAIL)\b'),                # Boolean literals

            # Arithmetic operations
            ('SUM_OF', r'\bSUM OF\b'),                   # Addition operation
            ('DIFF_OF', r'\bDIFF OF\b'),                 # Subtraction operation
            ('PRODUKT_OF', r'\bPRODUKT OF\b'),           # Multiplication operation
            ('QUOSHUNT_OF', r'\bQUOSHUNT OF\b'),         # Division operation
            ('MOD_OF', r'\bMOD OF\b'),                   # Modulo operation

            # Logical operations
            ('SMOOSH', r'\bSMOOSH\b'),                   # Concatenation operation

            # Assignment
            ('ASSIGN', r'\bR\b'),                        # Assignment operation
            ('A', r'\bA\b'),                             # 'A' keyword for MAEK operator
            # Operand separator
            ('AN', r'\bAN\b'),                           # Operand separator

            # Input/output
            ('VISIBLE', r'\bVISIBLE\b'),                 # Print statement keyword
            ('GIMMEH', r'\bGIMMEH\b'),                   # Input statement keyword

            # Flow-control statements
            ('YA_RLY', r'\bYA RLY\b'),                   # If clause
            ('NO_WAI', r'\bNO WAI\b'),                   # Else clause
            ('OIC', r'\bOIC\b'),                         # End of flow-control block

            # Looping
            ('IM_IN_YR', r'\bIM IN YR\b'),               # Start of loop
            ('IM_OUTTA_YR', r'\bIM OUTTA YR\b'),         # End of loop
            ('UPPIN', r'\bUPPIN\b'),                     # Increment operation
            ('NERFIN', r'\bNERFIN\b'),                   # Decrement operation
            ('TIL', r'\bTIL\b'),                         # Until condition
            ('WILE', r'\bWILE\b'),                       # While condition

            # Switch-case statements
            ('OMG', r'\bOMG\b'),                         # Case keyword
            ('OMGWTF', r'\bOMGWTF\b'),                   # Default case
            ('GTFO', r'\bGTFO\b'),                       # Break statement

            # Function definitions and calls
            ('HOW_IZ_I', r'\bHOW IZ I\b'),               # Function definition start
            ('I_IZ', r'\bI IZ\b'),                       # Function call
            ('FOUND_YR', r'\bFOUND YR\b'),               # Function return
            ('IF_U_SAY_SO', r'\bIF U SAY SO\b'),         # Function definition end

            ('WAZZUP', r'\bWAZZUP\b'),  # Start of a block
            ('BUHBYE', r'\bBUHBYE\b'),   # End of a block

            # Variable management
            ('I_HAS_A', r'\bI HAS A\b'),                 # Variable declaration
            ('ITZ', r'\bITZ\b'),                         # Variable initialization
            ('IT_KEYWORD', r'\bIT\b'),                   # Implicit IT variable
            ('VARIABLE_NAME', r'\b[a-zA-Z][a-zA-Z0-9_]*\b'),  # Variable names

            # Operators and special symbols
            ('PLUS', r'\+'),                             # Concatenation symbol for VISIBLE

            # Whitespace and structure
            ('NEWLINE', r'\n'),                          # Line separators
            ('WHITESPACE', r'[ \t]+'),                   # Skip over spaces and tabs

            # Error handling
            ('MISMATCH', r'.'),                          # Any other character
        ]

        # Compile the combined regex pattern with case-insensitive flag
        self.tok_regex = re.compile('|'.join(f'(?P<{name}>{pattern})' for name, pattern in token_specs), re.IGNORECASE)

    def tokenize(self, code):
        """Tokenizes source code."""
        tokens = []
        pos = 0
        line_number = 1
        nested_comment_level = 0
        end = len(code)

        while pos < end:
            match = self.tok_regex.match(code, pos)
            if match:
                kind = match.lastgroup
                value = match.group(kind)

                if kind == 'WHITESPACE':
                    pos = match.end()
                    continue

                if kind == 'NEWLINE':
                    pos = match.end()
                    line_number += 1
                    continue

                # Handle multi-line comments
                if nested_comment_level > 0:
                    if kind == 'TLDR':
                        nested_comment_level -= 1
                    pos = match.end()
                    continue

                if kind == 'OBTW':
                    nested_comment_level += 1
                    pos = match.end()
                    continue

                # Skip single-line comments
                if kind == 'single_line_comment':
                    pos = match.end()
                    continue

                # Handle mismatches
                if kind == 'MISMATCH':
                    print(f"Unexpected character at line {line_number}: {value!r}")
                    pos = match.end()
                    continue

                tokens.append((value, kind, line_number))
                pos = match.end()
            else:
                print(f"Unexpected character at position {pos}: {code[pos]!r}")
                pos += 1

        return tokens

# ============================
# Parser Class
# ============================

class Parser:
    def __init__(self, tokens, output_callback=None, input_callback=None):
        self.tokens = tokens
        self.pos = 0
        self.errors = []
        self.symbol_table = SymbolTable(output_callback=output_callback)  # Initialize SymbolTable with output callback
        self.output_callback = output_callback
        self.input_callback = input_callback

    def current_token(self):
        """Returns the current token."""
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def advance(self):
        """Advances to the next token."""
        self.pos += 1

    def lookahead_token(self, offset=1):
        """Returns the token ahead by the given offset without consuming it."""
        target_pos = self.pos + offset
        return self.tokens[target_pos] if target_pos < len(self.tokens) else None

    def match(self, expected_type):
        """
        Matches the current token with the expected type.
        Advances the position if matched.
        Raises CustomSyntaxError if not matched.
        """
        token = self.current_token()
        if token and token[1] == expected_type:
            self.advance()
            return token
        else:
            line_number = token[2] if token else "unknown"
            got = token[0] if token else "EOF"
            error_message = f"Expected {expected_type}, but got '{got}'. Error at line {line_number}"
            self.errors.append(error_message)
            raise CustomSyntaxError(error_message)

    def parse_program(self):
        """Parses the entire program."""
        self._parse_start()
        self.parse_code_block()
        self._parse_end()
        self._parse_post_end()

    def _parse_start(self):
        """Parses the start of the program."""
        token = self.current_token()
        if not token or token[1] != 'HAI':
            raise CustomSyntaxError("Program must start with 'HAI'.")
        self.match('HAI')  # Match 'HAI'

    def _parse_end(self):
        """Parses the end of the program."""
        token = self.current_token()
        if not token or token[1] != 'KTHXBYE':
            raise CustomSyntaxError("Program must end with 'KTHXBYE'.")
        self.match('KTHXBYE')  # Match 'KTHXBYE'

    def _parse_post_end(self):
        """Handles tokens after 'KTHXBYE'."""
        while self.current_token():
            token = self.current_token()
            if token[1] in ['HOW_IZ_I', 'I_IZ', 'OBTW', 'single_line_comment']:
                if token[1] == 'HOW_IZ_I':
                    self.parse_function_definition()
                elif token[1] == 'I_IZ':
                    self.parse_function_call()
                elif token[1] in ['OBTW', 'single_line_comment']:
                    self.parse_comment()
            else:
                raise CustomSyntaxError(f"Unexpected token after 'KTHXBYE': '{token[0]}' at line {token[2]}")

    def parse_code_block(self):
        """Parses a block of code with various statements."""
        while self.current_token():
            token = self.current_token()

            if token[1] == 'KTHXBYE':
                break  # End of program

            if token[1] == 'WAZZUP':
                self.match('WAZZUP')
                self.parse_statement()

            elif token[1] == 'BUHBYE':
                self.match('BUHBYE')

            elif token[1] == 'VARIABLE_NAME':
                next_token = self.lookahead_token()
                if next_token and next_token[1] in ['ASSIGN', 'IS_NOW_A']:
                    self.parse_assignment()
                else:
                    self.parse_expression()

            elif token[1] in ['I_HAS_A', 'VISIBLE', 'GIMMEH', 'O_RLY', 'WTF', 'IM_IN_YR',
                              'BOTH_SAEM', 'DIFFRINT', 'BIGGR_OF', 'SMALLR_OF',
                              'ALL_OF', 'ANY_OF', 'NOT', 'MAEK', 'SMOOSH', 'PLUS']:
                self.parse_statement()

            else:
                error_message = f"Unexpected statement: '{token[0]}' at line {token[2]}"
                self.errors.append(error_message)
                raise CustomSyntaxError(error_message)

    def parse_statement(self):
        """Parses different types of statements."""
        token = self.current_token()
        kind = token[1]

        statementhandlers = {
            'I_HAS_A': self.parse_var_declaration,
            'VISIBLE': self.parse_visible_statement,
            'GIMMEH': self.parse_input_statement,
            'O_RLY': self.parse_flow_control,
            'WTF': self.parse_switch_case,
            'IM_IN_YR': self.parse_loop_statement,
            'MAEK': self.parse_typecasting,
            'SMOOSH': self.parse_expression,
            'PLUS': self.parse_expression,
            'BOTH_SAEM': self.parse_expression,
            'DIFFRINT': self.parse_expression,
            'BIGGR_OF': self.parse_expression,
            'SMALLR_OF': self.parse_expression,
            'ALL_OF': self.parse_expression,
            'ANY_OF': self.parse_expression,
            'NOT': self.parse_expression,
        }

        handler = statementhandlers.get(kind, self.parse_expression)
        handler()

    def parse_var_declaration(self):
        """Parses a variable declaration."""
        self.match('I_HAS_A')
        var_token = self.match('VARIABLE_NAME')
        var_name = var_token[0]
        line_number = var_token[2]

        # Check for initialization
        if self.current_token() and self.current_token()[1] == 'ITZ':
            self.match('ITZ')
            expr_type, expr_value = self.parse_expression(return_type=True)
            var_type = expr_type
            self.symbol_table.declare_variable(var_name, var_type, line_number=line_number)
            self.symbol_table.assign_variable(var_name, var_type, expr_value, line_number=line_number)
        else:
            self.symbol_table.declare_variable(var_name, line_number=line_number)

    def parse_assignment(self):
        """Parses an assignment statement."""
        var_token = self.match('VARIABLE_NAME')
        var_name = var_token[0]
        line_number = var_token[2]

        if self.current_token() and self.current_token()[1] == 'IS_NOW_A':
            self.match('IS_NOW_A')
            type_token = self.match('TYPE')
            cast_type = type_token[0].upper()  # Get the new type to cast to
            
            # We don't need to parse a new expression, we simply re-cast the variable
            current_value = self.symbol_table.variables[var_name]['value']
            current_type = self.symbol_table.variables[var_name]['type']
            
            # Recast the variable using the current value and the new type
            self.symbol_table.assign_variable(var_name, cast_type, current_value, line_number=line_number)
            return

        self.match('ASSIGN')

        if self.current_token() and self.current_token()[1] == 'SMOOSH':
            self.match('SMOOSH')
            operands = self.parse_smoosh_operands()
            concatenated_value = ''.join(operands)
            self.symbol_table.assign_variable(var_name, 'YARN', concatenated_value, line_number=line_number)
            return

        if self.current_token() and self.current_token()[1] == 'MAEK':
            # Handle the 'MAEK' typecasting in an assignment
            self.match('MAEK')
            expr_type, expr_value = self.parse_expression(return_type=True) 
            self.match('A')  # 'A' is the separator for 'MAEK' expression and type

            # Get the target type for casting
            type_token = self.match('TYPE')
            cast_type = type_token[0].upper()
            
            # Perform the casting and assign the result to the variable
            casted_value = self.symbol_table._cast_value(expr_type, cast_type, expr_value)
            # Assign the casted value to the variable
            self.symbol_table.assign_variable(var_name, cast_type, casted_value, line_number=line_number)
            return

        # Standard assignment
        expr_type, expr_value = self.parse_expression(return_type=True)
         # If the type is `TROOF`, assign the value as 'WIN' or 'FAIL'
        if expr_type == 'TROOF':
            if expr_value == 'WIN':
                self.symbol_table.assign_variable(var_name, 'TROOF', 'WIN', line_number=line_number)
            elif expr_value == 'FAIL':
                self.symbol_table.assign_variable(var_name, 'TROOF', 'FAIL', line_number=line_number)
            else:
                self.symbol_table.assign_variable(var_name, 'TROOF', 'FAIL', line_number=line_number)
        else:
            self.symbol_table.assign_variable(var_name, expr_type, expr_value, line_number=line_number)

    def parse_smoosh_operands(self):
        """Parses operands for the SMOOSH operation."""
        operands = []
        while True:
            expr_type, expr_value = self.parse_expression(return_type=True)
            if expr_type != 'YARN':
                raise SemanticError(f"SMOOSH operation requires YARN types, got {expr_type}")
            operands.append(expr_value)

            if self.current_token() and self.current_token()[1] == 'AN':
                self.match('AN')
            else:
                break
        return operands

    def parse_flow_control(self):
        """Parses flow-control statements (O RLY?)."""
        self.match('O_RLY')
        self.match('YA_RLY')
        
        # Get the condition from the symbol table (IT variable)
        condition = self.symbol_table.get_it_value()
        
        if condition == 'WIN':
            self.parse_code_block()  # Execute YA RLY block
        else:
            # Skip YA RLY block
            while self.current_token() and self.current_token()[1] not in ['MEBBE', 'NO_WAI', 'OIC']:
                self.advance()

        # Handle MEBBE (else if) blocks
        while self.current_token() and self.current_token()[1] == 'MEBBE':
            self.match('MEBBE')
            expr_type, expr_value = self.parse_expression(return_type=True)
            
            if condition == 'FAIL' and expr_value == 'WIN':
                self.parse_code_block()  # Execute MEBBE block
                condition = 'WIN'  # Mark that we've executed a block
            else:
                # Skip this MEBBE block
                while self.current_token() and self.current_token()[1] not in ['MEBBE', 'NO_WAI', 'OIC']:
                    self.advance()

        # Handle NO WAI (else) block
        if self.current_token() and self.current_token()[1] == 'NO_WAI':
            self.match('NO_WAI')
            if condition == 'FAIL':
                self.parse_code_block()  # Execute NO WAI block
            else:
                # Skip NO WAI block
                while self.current_token() and self.current_token()[1] != 'OIC':
                    self.advance()

        self.match('OIC')



    def parse_switch_case(self):
        """Parses switch-case statements (WTF?)."""
        self.match('WTF')
        expr_type, expr_value = self.parse_expression(return_type=True)  # Get the value of IT

        while self.current_token() and self.current_token()[1] == 'OMG':
            self.match('OMG')
            case_value = self.match('VARIABLE_NAME')[0]  # Get the literal value in OMG
            if case_value == expr_value:
                self.parse_code_block()  # Execute the block if the value matches
                self.match('GTFO')  # End the case block if matched
                break
            else:
                self.match('GTFO')  # Skip this block if no match

        # Default case
        if self.current_token() and self.current_token()[1] == 'OMGWTF':
            self.match('OMGWTF')
            self.parse_code_block()  # Default block (if no match)

        self.match('OIC')  # End of switch-case block



    def parse_loop_statement(self):
        """Parses loop statements (IM IN YR)."""
        self.match('IM_IN_YR')
        label_token = self.match('VARIABLE_NAME')  # Read label (unused for now)
        self.match('YR')
        self.parse_code_block()

        # Handle iteration (UPPIN, NERFIN)
        if self.current_token() and self.current_token()[1] in ['UPPIN', 'NERFIN']:
            op = self.match('UPPIN') if self.current_token()[1] == 'UPPIN' else self.match('NERFIN')
            var_token = self.match('VARIABLE_NAME')
            var_name = var_token[0]
            current_value = self.symbol_table.variables[var_name]['value']
            increment = 1 if op == 'UPPIN' else -1
            self.symbol_table.assign_variable(var_name, 'NUMBR', current_value + increment)

        # Handle TIL (continue until condition) or WILE (continue while condition)
        if self.current_token() and self.current_token()[1] in ['TIL', 'WILE']:
            condition = self.parse_expression(return_type=True)[1]  # Evaluate condition
            if self.current_token()[1] == 'TIL' and condition == 'WIN':
                self.match('TIL')
            elif self.current_token()[1] == 'WILE' and condition == 'FAIL':
                self.match('WILE')

        self.match('IM_OUTTA_YR')  # End of loop block



    def parse_function_definition(self):
        """Parses function definitions (HOW IZ I)."""
        self.match('HOW_IZ_I')
        func_token = self.match('VARIABLE_NAME')
        func_name = func_token[0]
        self.output_callback(f"Function '{func_name}' defined at line {func_token[2]}.")
        # Parse function parameters (if any)
        while self.current_token() and self.current_token()[1] == 'YR':
            self.match('YR')
            self.match('VARIABLE_NAME')

        self.match('IF_U_SAY_SO')  # End of function definition



    def parse_function_call(self):
        """Parses function calls (I IZ)."""
        self.match('I_IZ')
        func_token = self.match('VARIABLE_NAME')
        func_name = func_token[0]
        self.output_callback(f"Function '{func_name}' called at line {func_token[2]}.")
        # Parse function arguments
        args = []
        while self.current_token() and self.current_token()[1] == 'YR':
            self.match('YR')
            expr_type, expr_value = self.parse_expression(return_type=True)
            args.append((expr_type, expr_value))
        
        self.match('MKAY')  # End of function call

    def parse_return(self):
        """Parses a return statement (FOUND YR)."""
        self.match('FOUND_YR')
        expr_type, expr_value = self.parse_expression(return_type=True)
        self.output_callback(f"Returning value '{expr_value}' of type '{expr_type}'")
        return expr_type, expr_value


    def parse_comment(self):
        """Parses and skips comments."""
        token = self.current_token()
        if token[1] == 'OBTW':
            self.match('OBTW')
            while self.current_token() and self.current_token()[1] != 'TLDR':
                self.advance()
            self.match('TLDR')
        elif token[1] == 'single_line_comment':
            self.match('single_line_comment')

    def parse_visible_statement(self):
        """Parses a VISIBLE statement."""
        self.match('VISIBLE')
        output_str = self._build_output()

        self.symbol_table._log(output_str)

    def _build_output(self):
        """Helper method to build output string for VISIBLE."""
        output_str = ""
        expr_type, expr_value = self.parse_expression(return_type=True)
        output_str += str(expr_value)

        while self.current_token() and self.current_token()[1] in ['AN', 'PLUS']:
            operator = self.current_token()[1]
            self.match(operator)
            next_expr_type, next_expr_value = self.parse_expression(return_type=True)
            output_str += str(next_expr_value)

        return output_str

    def parse_input_statement(self):
        """Parses a GIMMEH statement."""
        self.match('GIMMEH')
        var_token = self.match('VARIABLE_NAME')
        variable = var_token[0]
        line_number = var_token[2]

        user_input = self.input_callback(variable) if self.input_callback else "NOOB"

        var_type = self.symbol_table.get_variable_type(variable, line_number=line_number)
        user_input, var_type = self._process_input(user_input, var_type, variable, line_number)

        self.symbol_table.assign_variable(variable, var_type, user_input, line_number=line_number)
        self.symbol_table._log(f"Input assigned to variable '{variable}' with type '{var_type}' and value '{user_input}'.")

    def _process_input(self, user_input, var_type, variable, line_number):
        """Processes and validates user input based on variable type."""
        try:
            if var_type == 'NOOB':
                user_input, var_type = self.infer_type(user_input)
            elif var_type == 'NUMBR':
                user_input = int(user_input)
            elif var_type == 'NUMBAR':
                user_input = float(user_input)
            elif var_type == 'TROOF':
                if user_input.upper() == 'WIN':
                    user_input = True
                elif user_input.upper() == 'FAIL':
                    user_input = False
                else:
                    raise SemanticError(f"Invalid input for TROOF variable '{variable}'. Expected 'WIN' or 'FAIL'.")
            elif var_type == 'YARN':
                user_input = user_input.strip('"')  # Remove quotes if necessary
        except ValueError:
            raise SemanticError(f"Invalid input for type '{var_type}' for variable '{variable}'.")
        return user_input, var_type

    def infer_type(self, user_input):
        """Infer the type of user input and return the value and type."""
        user_input = user_input.strip('"')  # Remove quotes if present
        if re.fullmatch(r'-?\d+', user_input):
            return int(user_input), 'NUMBR'
        elif re.fullmatch(r'-?\d+\.\d+', user_input):
            return float(user_input), 'NUMBAR'
        elif user_input.upper() in ['WIN', 'FAIL']:
            return True if user_input.upper() == 'WIN' else False, 'TROOF'
        else:
            return user_input, 'YARN'

    
    def parse_typecasting(self):
        """Parses explicit typecasting using MAEK."""
        self.match('MAEK')  # Match the 'MAEK' keyword
        expr_type, expr_value = self.parse_expression(return_type=True)  # Parse the expression
        
        # Check if 'A' keyword is present (it's optional in some cases)
        if self.current_token() and self.current_token()[1] == 'A':
            self.match('A')
        
        # Get the desired type
        type_token = self.match('TYPE')
        cast_type = type_token[0].upper()  # The target type to cast to
        
        # Perform the casting based on the target type
        casted_value = self._cast_value(expr_type, cast_type, expr_value)
        
        return cast_type, casted_value


    def parse_expression(self, return_type=False):
        """Parses an expression and optionally returns its type and value."""
        token = self.current_token()
        if not token:
            if return_type:
                return None, None
            return

        kind = token[1]
        value = token[0]

        expr_type = None
        expr_value = None

        expressionhandlers = {
            'VARIABLE_NAME': self.handle_variable,
            'NUMBR': self.handle_numbr,
            'NUMBAR': self.handle_numbar,
            'YARN': self.handle_yarn,
            'TROOF': self.handle_troof,
            'SUM_OF': self.handle_arithmetic,
            'DIFF_OF': self.handle_arithmetic,
            'PRODUKT_OF': self.handle_arithmetic,
            'QUOSHUNT_OF': self.handle_arithmetic,
            'MOD_OF': self.handle_arithmetic,
            'BOTH_SAEM': self.handle_comparison,
            'DIFFRINT': self.handle_comparison,
            'BIGGR_OF': self.handle_relational,
            'SMALLR_OF': self.handle_relational,
            'BOTH_OF': self.handle_logical,
            'EITHER_OF': self.handle_logical,
            'WON_OF': self.handle_logical,
            'NOT': self.handle_not,
            'SMOOSH': self.handle_smoosh,
            'PLUS': self.handle_plus,
            'ANY_OF': self.handle_any_of,
            'ALL_OF': self.handle_all_of,
            'MAEK': self.handle_maek,
        }

        handler = expressionhandlers.get(kind, self.handle_unexpected)
        result = handler()

        if return_type:
            return result

        return

    def handle_variable(self):
        """Handles variable usage in expressions."""
        var_token = self.match('VARIABLE_NAME')
        var_name = var_token[0]
        line_number = var_token[2]
        var_type = self.symbol_table.get_variable_type(var_name, line_number=line_number)
        self.symbol_table.is_initialized(var_name, line_number=line_number)
        var_value = self.symbol_table.variables[var_name]['value']
        var_value = 'NOOB' if var_type == 'NOOB' else var_value
        return var_type, var_value

    def handle_numbr(self):
        """Handles NUMBR literals."""
        token = self.match('NUMBR')
        return 'NUMBR', int(token[0])

    def handle_numbar(self):
        """Handles NUMBAR literals."""
        token = self.match('NUMBAR')
        return 'NUMBAR', float(token[0])

    def handle_yarn(self):
        """Handles YARN literals."""
        token = self.match('YARN')
        return 'YARN', token[0].strip('"')

    def handle_troof(self):
        """Handles TROOF literals."""
        token = self.match('TROOF')
        value = 'WIN' if token[0].upper() == 'WIN' else 'FAIL'
        return 'TROOF', value

    def handle_arithmetic(self):
        """Handles arithmetic operations."""
        operator = self.current_token()[1]
        self.match(operator)
        left_type, left_value = self.parse_expression(return_type=True)
        self.match('AN')
        right_type, right_value = self.parse_expression(return_type=True)

        # Added type conversion for TROOF :> eto yung kulang sa handle arithmetic
        if left_type in ['TROOF', 'WIN', 'FAIL']:
            if left_value.upper() == 'WIN':
                left_value = 1
            elif left_value.upper() == 'FAIL':
                left_value = 0
            left_type in ['NUMBR', 'NUMBAR']

        if right_type in ['TROOF', 'WIN', 'FAIL']:
            if right_value.upper() == 'WIN':
                right_value = 1
            elif right_value.upper() == 'FAIL':
                right_value = 0
            right_type in ['NUMBR', 'NUMBAR']
            
        # Implicit casting
        if left_type == 'YARN' and right_type in ['NUMBR', 'NUMBAR']:
            left_value = float(left_value) if left_type == 'NUMBAR' else int(left_value)
            left_type = right_type
        elif right_type == 'YARN' and left_type in ['NUMBR', 'NUMBAR']:
            right_value = float(right_value) if right_type == 'NUMBAR' else int(right_value)
            right_type = left_type

        # Determine result type
        if operator in ['SUM_OF', 'DIFF_OF', 'PRODUKT_OF', 'QUOSHUNT_OF', 'MOD_OF']:
            if 'NUMBAR' in [left_type, right_type]:
                result_type = 'NUMBAR'
                left_value = float(left_value)
                right_value = float(right_value)
            else:
                result_type = 'NUMBR'
                left_value = int(left_value)
                right_value = int(right_value)
        else:
            result_type = 'UNKNOWN'

        
            
        # Perform operation
        operations = {
            'SUM_OF': left_value + right_value,
            'DIFF_OF': left_value - right_value,
            'PRODUKT_OF': left_value * right_value,
            'QUOSHUNT_OF': self.handle_division(left_type, left_value, right_type, right_value),
            'MOD_OF': self.handle_modulo(left_type, left_value, right_type, right_value),
        }

        expr_value = operations.get(operator, None)
        return result_type, expr_value

    def handle_division(self, left_type, left_value, right_type, right_value):
        """Handles division operation, allowing division by zero."""
        if right_value == 0:
            if left_type == 'NUMBAR':
                return float('inf') if left_value >= 0 else float('-inf')
            elif left_type == 'NUMBR':
                return 'INF' if left_value >= 0 else '-INF'
        return left_value / right_value

    def handle_modulo(self, left_type, left_value, right_type, right_value):
        """Handles modulo operation, allowing modulo by zero."""
        if right_value == 0:
            if left_type == 'NUMBAR':
                return float('nan')  # Not a Number
            elif left_type == 'NUMBR':
                return 'UNDEFINED'  # Custom representation
        return left_value % right_value

    def handle_comparison(self):
        """Handles comparison operations."""
        operator = self.current_token()[1]
        self.match(operator)
        left_type, left_value = self.parse_expression(return_type=True)
        self.match('AN')
        right_type, right_value = self.parse_expression(return_type=True)

        if left_type != right_type:
            raise SemanticError(f"Comparison requires both operands to be of the same type, got '{left_type}' and '{right_type}'.")

        result_type = 'TROOF'
        result_value = left_value == right_value if operator == 'BOTH_SAEM' else left_value != right_value
        return result_type, result_value

    def handle_relational(self):
        """Handles relational operations."""
        operator = self.current_token()[1]
        self.match(operator)
        left_type, left_value = self.parse_expression(return_type=True)
        self.match('AN')
        right_type, right_value = self.parse_expression(return_type=True)

        if left_type != right_type:
            raise SemanticError(f"Relational operation '{operator}' requires both operands to be of the same type, got '{left_type}' and '{right_type}'.")

        if operator == 'BIGGR_OF':
            expr_value = max(left_value, right_value)
        elif operator == 'SMALLR_OF':
            expr_value = min(left_value, right_value)
        else:
            expr_value = None  # Undefined operator

        return left_type, expr_value

    def handle_all_of(self):
        """Handles the ALL OF operation."""
        self.match('ALL_OF')  # Match the ALL OF token
        operands = []

        # Parse each operand in the ALL OF expression
        while self.current_token() and self.current_token()[1] != 'MKAY':
            operand_type, operand_value = self.parse_expression(return_type=True)
            
            # Treat '0' as 'FAIL' and any non-zero as 'WIN'
            if operand_value == 0 or operand_value == '0':
                operand_value = 'FAIL'
            elif operand_value == 1 or operand_value == '1' or operand_value == 'WIN':
                operand_value = 'WIN'
            
            # Ensure the operand is of type TROOF (boolean)
            if operand_value not in ['WIN', 'FAIL']:
                raise SemanticError(f"ALL OF operation requires TROOF types, got '{operand_value}'.")

            operands.append(operand_value)

            if self.current_token() and self.current_token()[1] == 'AN':
                self.match('AN')  # Match 'AN' and continue parsing
            else:
                break

        # Ensure the operation ends with MKAY
        self.match('MKAY')

        # Perform the logical AND (ALL OF) operation
        expr_value = 'WIN' if all(operand == 'WIN' for operand in operands) else 'FAIL'
        
        return 'TROOF', expr_value


    def handle_any_of(self):
        """Handles the ANY OF operation."""
        self.match('ANY_OF')  # Match the ANY OF token
        operands = []

        # Parse each operand in the ANY OF expression
        while self.current_token()[1] != 'MKAY':  # Keep parsing until we hit MKAY
            operand_type, operand_value = self.parse_expression(return_type=True)
            
            # Treat '0' as 'FAIL' and any non-zero as 'WIN'
            if operand_value == 0 or operand_value == '0':
                operand_value = 'FAIL'
            elif operand_value == 1 or operand_value == '1' or operand_value == 'WIN':
                operand_value = 'WIN'
            
            # Ensure the operand is of type TROOF (boolean)
            if operand_value not in ['WIN', 'FAIL']:
                raise SemanticError(f"ANY OF operation requires TROOF types, got '{operand_value}'.")

            operands.append(operand_value)
            if self.current_token()[1] == 'AN':
                self.match('AN')  # Match 'AN' and continue parsing
            else:
                break

        self.match('MKAY')
        
        # Perform the logical OR (ANY OF) operation
        expr_value = 'WIN' if any(operand == 'WIN' for operand in operands) else 'FAIL'
        
        return 'TROOF', expr_value


    def parse_operands(self):
        """Parse operands for logical operations."""
        operands = []
        while self.current_token() and self.current_token()[1] not in ['MKAY', 'AN']:
            operand_type, operand_value = self.parse_expression(return_type=True)
            
            # Check for invalid nested logical operations
            if operand_type == 'LOGICAL':
                if operand_value in ['ALL_OF', 'ANY_OF']:
                    raise SemanticError("Nested ALL_OF or ANY_OF is not allowed.")
            
            operands.append((operand_type, operand_value))
            
        return operands
    
    
    
    def handle_logical(self):
        """Handles logical operations."""
        operator = self.current_token()[1]
        self.match(operator)
        left_type, left_value = self.parse_expression(return_type=True)
        self.match('AN')
        right_type, right_value = self.parse_expression(return_type=True)

        if left_type != 'TROOF' or right_type != 'TROOF':
            raise SemanticError(f"Logical operations require TROOF types, got '{left_type}' and '{right_type}'.")

        # Perform the logical operations and return 'WIN' or 'FAIL' instead of boolean
        if operator == 'BOTH_OF':
            expr_value = 'WIN' if left_value and right_value else 'FAIL'
        elif operator == 'EITHER_OF':
            expr_value = 'WIN' if left_value or right_value else 'FAIL'
        elif operator == 'WON_OF':
            expr_value = 'WIN' if left_value != right_value else 'FAIL'
        else:
            raise SemanticError(f"Unknown logical operator: '{operator}'")

        return 'TROOF', expr_value


    def handle_not(self):
        """Handles NOT operation."""
        self.match('NOT')
        operand_type, operand_value = self.parse_expression(return_type=True)

        if operand_type != 'TROOF':
            raise SemanticError(f"NOT operation requires TROOF type, got '{operand_type}'.")

        # Perform the NOT operation and return 'WIN' if True, 'FAIL' if False
        expr_value = 'WIN' if not operand_value else 'FAIL'
        
        return 'TROOF', expr_value


    def handle_smoosh(self):
        """Handles SMOOSH operation for string concatenation."""
        self.match('SMOOSH')
        operands = self.parse_smoosh_operands()
        return 'YARN', ''.join(operands)

    def handle_maek(self):
        # MAEK <expression> [A] <type>
        self.match('MAEK')
        expr_type, expr_value = self.parse_expression(return_type=True)

        # The 'A' is optional
        if self.current_token() and self.current_token()[1] == 'A':
            self.match('A')

        type_token = self.match('TYPE')
        cast_type = type_token[0].upper()

        casted_value = self.cast_value(expr_value, expr_type, cast_type)
        return cast_type, casted_value

    def handle_plus(self):
        """Handles PLUS operator for string concatenation or addition."""
        self.match('PLUS')
        left_type, left_value = self.parse_expression(return_type=True)
        right_type, right_value = self.parse_expression(return_type=True)

        # Implicit casting for YARN and NUMBR/NUMBAR
        if left_type == 'YARN' and right_type in ['NUMBR', 'NUMBAR']:
            right_value = str(right_value)
            right_type = 'YARN'
        elif right_type == 'YARN' and left_type in ['NUMBR', 'NUMBAR']:
            left_value = str(left_value)
            left_type = 'YARN'

        if left_type != right_type:
            raise SemanticError(f"PLUS operation requires matching types, got '{left_type}' and '{right_type}'.")

        if left_type in ['NUMBR', 'NUMBAR']:
            return left_type, left_value + right_value
        elif left_type == 'YARN':
            return 'YARN', left_value + right_value
        else:
            raise SemanticError(f"PLUS operation not supported for type '{left_type}'.")

    def handle_unexpected(self):
        """Handles unexpected tokens in expressions."""
        token = self.current_token()
        raise CustomSyntaxError(f"Unexpected token in expression: '{token[0]}' at line {token[2]}")

# ============================
# GUI Application Class
# ============================

class GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("LOL IDE")
        self.root.geometry("1200x800")

        # Configure grid layout
        self.root.rowconfigure(0, weight=1)  # Top frames
        self.root.rowconfigure(1, weight=0)  # Execute button
        self.root.rowconfigure(2, weight=1)  # Console
        self.root.columnconfigure(0, weight=1)

        # Create top frames (three side by side)
        self.top_frame = ttk.Frame(self.root)
        self.top_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.top_frame.columnconfigure((0, 1, 2), weight=1)
        self.top_frame.rowconfigure(0, weight=1)

        # Initialize all sub-frames
        self._init_file_editor_frame()
        self._init_tokens_frame()
        self._init_symbol_table_frame()

        # Initialize Execute Button and Console
        self._init_execute_button()
        self._init_console()

    def _init_file_editor_frame(self):
        """Initializes the File Explorer & Text Editor frame."""
        self.file_editor_frame = ttk.LabelFrame(self.top_frame, text="File Explorer & Text Editor")
        self.file_editor_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.file_editor_frame.columnconfigure((0, 1), weight=1)
        self.file_editor_frame.rowconfigure(1, weight=1)

        # File Explorer Button
        self.browse_button = ttk.Button(self.file_editor_frame, text="Browse File", command=self.browse_file)
        self.browse_button.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

        

        # Text Editor
        self.text_editor = ScrolledText(self.file_editor_frame, wrap=tk.WORD)
        self.text_editor.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)

    def _init_tokens_frame(self):
        """Initializes the List of Tokens frame."""
        self.tokens_frame = ttk.LabelFrame(self.top_frame, text="List of Tokens")
        self.tokens_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        self.tokens_frame.rowconfigure(0, weight=1)
        self.tokens_frame.columnconfigure(0, weight=1)

        # Treeview for Tokens
        self.tokens_tree = ttk.Treeview(self.tokens_frame, columns=("Lexeme", "Classification"), show='headings')
        self.tokens_tree.heading("Lexeme", text="Lexeme")
        self.tokens_tree.heading("Classification", text="Classification")
        self.tokens_tree.column("Lexeme", width=150, anchor='center')
        self.tokens_tree.column("Classification", width=150, anchor='center')
        self.tokens_tree.grid(row=0, column=0, sticky="nsew")

        # Scrollbar for Tokens
        self.tokens_scrollbar = ttk.Scrollbar(self.tokens_frame, orient="vertical", command=self.tokens_tree.yview)
        self.tokens_tree.configure(yscroll=self.tokens_scrollbar.set)
        self.tokens_scrollbar.grid(row=0, column=1, sticky='ns')

    def _init_symbol_table_frame(self):
        """Initializes the Symbol Table frame."""
        self.symbol_table_frame = ttk.LabelFrame(self.top_frame, text="Symbol Table")
        self.symbol_table_frame.grid(row=0, column=2, sticky="nsew", padx=5, pady=5)
        self.symbol_table_frame.rowconfigure(0, weight=1)
        self.symbol_table_frame.columnconfigure(0, weight=1)

        # Treeview for Symbol Table
        self.symbol_tree = ttk.Treeview(self.symbol_table_frame, columns=("Identifier", "Type", "Value"), show='headings')
        self.symbol_tree.heading("Identifier", text="Identifier")
        self.symbol_tree.heading("Type", text="Type")
        self.symbol_tree.heading("Value", text="Value")
        self.symbol_tree.column("Identifier", width=100, anchor='center')
        self.symbol_tree.column("Type", width=100, anchor='center')
        self.symbol_tree.column("Value", width=100, anchor='center')
        self.symbol_tree.grid(row=0, column=0, sticky="nsew")

        # Scrollbar for Symbol Table
        self.symbol_scrollbar = ttk.Scrollbar(self.symbol_table_frame, orient="vertical", command=self.symbol_tree.yview)
        self.symbol_tree.configure(yscroll=self.symbol_scrollbar.set)
        self.symbol_scrollbar.grid(row=0, column=1, sticky='ns')

    def _init_execute_button(self):
        """Initializes the Execute/Run button."""
        self.execute_frame = ttk.Frame(self.root)
        self.execute_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        self.execute_frame.columnconfigure(0, weight=1)

        self.run_button = ttk.Button(self.execute_frame, text="Execute/Run", command=self.execute_code)
        self.run_button.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

    def _init_console(self):
        """Initializes the Console frame."""
        self.console_frame = ttk.LabelFrame(self.root, text="Console")
        self.console_frame.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)
        self.console_frame.rowconfigure(0, weight=1)
        self.console_frame.columnconfigure(0, weight=1)

        self.console = ScrolledText(self.console_frame, wrap=tk.WORD, state='disabled')
        self.console.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

    def browse_file(self):
        """Browse and load a file into the text editor."""
        file_path = filedialog.askopenfilename(
            title="Select LOL Code File",
            filetypes=[("LOL Code Files", "*.lol"), ("All Files", "*.*")]
        )
        if file_path:
            try:
                with open(file_path, 'r') as file:
                    code = file.read()
                self.text_editor.delete(1.0, tk.END)
                self.text_editor.insert(tk.END, code)
                self.log_console(f"Loaded file: {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file:\n{e}")


    def execute_code(self):
        """Execute the code present in the text editor."""
        code = self.text_editor.get(1.0, tk.END).strip()
        if not code:
            self.log_console("No code to execute.")
            return

        self.clear_all()

        lexer = LexicalAnalyzer()
        tokens = lexer.tokenize(code)

        # Populate tokens treeview
        for lexeme, classification, _ in tokens:
            self.tokens_tree.insert('', tk.END, values=(lexeme, classification))

        # Define output and input callbacks
        output_callback = self.log_console
        input_callback = self.get_user_input

        parser = Parser(tokens, output_callback=output_callback, input_callback=input_callback)
        try:
            parser.parse_program()
            self.log_console("Parsing successful.")
        except (CustomSyntaxError, SemanticError) as e:
            self.log_console(f"Error: {e}")

        # Populate symbol table treeview
        for var, info in parser.symbol_table.variables.items():
            var_type = info['type']
            var_value = info['value']
            var_value_display = 'NOOB' if var_type == 'NOOB' else var_value
            self.symbol_tree.insert('', tk.END, values=(var, var_type, var_value_display))

    def get_user_input(self, variable):
        """Prompts the user for input via a dialog."""
        user_input = simpledialog.askstring("Input Required", f"Enter value for '{variable}':", parent=self.root)
        return user_input if user_input is not None else "NOOB"

    def log_console(self, message):
        """Logs a message to the console."""
        self.console.configure(state='normal')
        self.console.insert(tk.END, message + '\n')
        self.console.configure(state='disabled')
        self.console.see(tk.END)

    def clear_all(self):
        """Clears the console, tokens, and symbol table."""
        self.clear_console()
        self.clear_tokens()
        self.clear_symbol_table()

    def clear_console(self):
        """Clears the console."""
        self.console.configure(state='normal')
        self.console.delete(1.0, tk.END)
        self.console.configure(state='disabled')

    def clear_tokens(self):
        """Clears the tokens treeview."""
        for item in self.tokens_tree.get_children():
            self.tokens_tree.delete(item)

    def clear_symbol_table(self):
        """Clears the symbol table treeview."""
        for item in self.symbol_tree.get_children():
            self.symbol_tree.delete(item)

# ============================
# Main Execution
# ============================

def main():
    root = tk.Tk()
    app = GUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()