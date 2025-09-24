'''
Utility functions
'''

import os
import time
import random
import re
from typing import Dict, List, Optional, Union
from openai import OpenAI
import config


def setup_client() -> OpenAI:
    """Setup OpenAI client"""
    return OpenAI(
        api_key=config.OPENAI_API_KEY,
        base_url=config.OPENAI_BASE_URL,
    )


def get_response(client: Union[OpenAI, None], messages: List[Dict], model: str, temperature: float = 1,
                 top_p: float = 1, max_retries: int = 5, base_delay: float = 1.0,
                 enable_cot: bool = False, default: bool = False) -> Dict[str, str]:
    """Call LLM to get response with exponential backoff retry mechanism and CoT support

    Args:
        client: OpenAI client
        messages: Message list
        model: Model name or path
        temperature: Temperature parameter
        top_p: top_p parameter
        max_retries: Maximum retry attempts
        base_delay: Base delay time
        enable_cot: Whether to enable CoT chain of thought
        default: Whether to use default parameters

    Returns:
        Dict containing content and reasoning_content (if any)
    """
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            # 根据模型类型构建请求参数 / Build request parameters based on model type
            request_params = {
                "model": model,
                "messages": messages,
                "max_completion_tokens": 4096
            }
            
            # 如果不使用默认参数，则添加temperature和top_p / Add temperature and top_p if not using default parameters
            if not default:
                request_params["temperature"] = temperature
                request_params["top_p"] = top_p
            
            # 如果启用CoT，根据模型类型添加相应参数 / Add corresponding parameters based on model type if CoT is enabled
            if enable_cot:
                if "claude" in model.lower():
                    # Claude模型使用thinking参数 / Claude models use thinking parameter
                    request_params["extra_body"] = {
                        "extendParams": {
                            "thinking": {
                                "type": "enabled",
                                "budget_tokens": 1024
                            }
                        }
                    }
                elif "gemini" in model.lower():
                    # Gemini模型使用thinkingConfig参数 / Gemini models use thinkingConfig parameter
                    request_params["extra_body"] = {
                        "extendParams": {
                            "thinkingConfig": {
                                "includeThoughts": 'true'
                            }
                        }
                    }

            completion = client.chat.completions.create(**request_params)
            
            # 提取响应内容 / Extract response content
            result = {
                "content": completion.choices[0].message.content,
                "reasoning_content": ""
            }
            
            # 如果启用了CoT且有reasoning_content，提取思维链内容 / Extract reasoning content if CoT is enabled and reasoning_content exists
            if enable_cot and hasattr(completion.choices[0].message, 'reasoning_content'):
                reasoning_content = completion.choices[0].message.reasoning_content
                if reasoning_content:
                    result["reasoning_content"] = reasoning_content
                else:
                    print("\nCoT not found")
            return result
            
        except Exception as e:
            last_exception = e
            error_message = str(e)
            print(f"Model call failed (attempt {attempt + 1}/{max_retries}): {error_message}")
            
            # 如果是最后一次尝试，不需要等待 / No need to wait if it's the last attempt
            if attempt == max_retries - 1:
                break
            
            # 检查是否是限流错误 / Check if it's a rate limit error
            is_rate_limit = any(keyword in error_message.lower() for keyword in ['429', 'rate', 'limit', 'throttle', 'quota', 'too many requests'])
            
            if is_rate_limit:
                # 对于限流错误，使用更长的等待时间 / Use longer wait time for rate limit errors
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                # 限流时最多等待60秒 / Wait up to 60 seconds for rate limiting
                delay = min(delay, 60.0)
                print(f"Rate limit error detected, waiting {delay:.1f} seconds before retry...")
            else:
                # 对于其他错误，使用较短的等待时间 / Use shorter wait time for other errors
                delay = base_delay * (1.5 ** attempt) + random.uniform(0, 0.5)
                delay = min(delay, 10.0)
                print(f"Waiting {delay:.1f} seconds before retry...")
            
            time.sleep(delay)
    
    print(f"All retries failed, last error: {last_exception}")
    return {"content": "", "reasoning_content": ""}


def prompt_from_file(path, **kwargs) -> str:
    """
    Read prompt template from file and format it into usable prompt

    Args:
        path (str): Prompt template file path
        **kwargs: Parameters for formatting template

    Returns:
        str: Formatted prompt
    """
    # 如果路径不是绝对路径，则相对于项目根目录 / If path is not absolute, make it relative to project root
    if not os.path.isabs(path):
        # 避免循环导入，直接计算项目根目录 / Avoid circular import, calculate project root directly
        project_root = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(project_root, path)
    
    with open(path, "r", encoding="utf-8") as file:
        content = file.read()
    
    # 如果提供了格式化参数，则进行格式化 / Format if formatting parameters are provided
    if kwargs:
        content = content.format(**kwargs)
    
    return content


def clean_json_response(response: str) -> str:
    """清理响应中的markdown代码块格式
    Clean markdown code block format from response

    Args:
        response: 原始响应字符串 / Original response string

    Returns:
        清理后的JSON字符串 / Cleaned JSON string
    """
    cleaned = response.strip()
    if cleaned.startswith("```json") and cleaned.endswith("```"):
        # 提取```json和```之间的内容 / Extract content between ```json and ```
        cleaned = cleaned[7:-3].strip()
    elif cleaned.startswith("```") and cleaned.endswith("```"):
        # 处理只有```的情况 / Handle case with only ```
        cleaned = cleaned[3:-3].strip()
    return cleaned


def extract_thinking_and_output(response: str) -> Dict[str, str]:
    """从响应中提取思考过程和最终输出
    Extract thinking process and final output from response

    根据论文，支持特定格式 / According to the paper, supports specific format:
    - <think>...</think> 和/and <output>...</output>
    """
    thinking_pattern = r'<think>(.*?)</think>'
    output_pattern = r'<output>(.*?)</output>'
    
    thinking_match = re.search(thinking_pattern, response, re.DOTALL)
    output_match = re.search(output_pattern, response, re.DOTALL)
    
    return {
        'thinking': thinking_match.group(1).strip() if thinking_match else "",
        'output': output_match.group(1).strip() if output_match else response.strip()
    }


def auto_adjust_excel_column_width(worksheet, dataframe):
    """自动调整Excel工作表的列宽和行高
    Auto-adjust Excel worksheet column width and row height

    Args:
        worksheet: openpyxl的工作表对象 / openpyxl worksheet object
        dataframe: pandas DataFrame
    """
    from openpyxl.styles import Alignment
    
    # 自动调整列宽 / Auto-adjust column width
    for column in dataframe:
        column_length = max(dataframe[column].astype(str).map(len).max(), len(column))
        col_idx = dataframe.columns.get_loc(column)
        # 获取Excel列名（支持超过26列的情况）/ Get Excel column name (supports more than 26 columns)
        col_letter = ''
        temp_idx = col_idx
        while temp_idx >= 0:
            col_letter = chr(65 + temp_idx % 26) + col_letter
            temp_idx = temp_idx // 26 - 1
        # 限制最大宽度为50 / Limit maximum width to 50
        worksheet.column_dimensions[col_letter].width = min(column_length + 2, 50)
    
    # 自动调整行高并设置文本换行 / Auto-adjust row height and set text wrapping
    for row_idx, row in enumerate(dataframe.itertuples(index=False), start=2):  # 从第2行开始（第1行是标题）/ Start from row 2 (row 1 is header)
        max_lines = 1
        for col_idx, cell_value in enumerate(row):
            # 获取单元格 / Get cell
            cell = worksheet.cell(row=row_idx, column=col_idx + 1)
            
            # 设置所有单元格默认顶端对齐 / Set all cells to default top alignment
            cell.alignment = Alignment(vertical='top')
            
            if isinstance(cell_value, str):
                # 计算单元格内容的行数（考虑换行符和超长文本）/ Calculate number of lines in cell content (considering line breaks and long text)
                lines = cell_value.count('\n') + 1
                # 也考虑过长的单行文本（假设50个字符后会自动换行）/ Also consider long single-line text (assuming auto-wrap after 50 characters)
                for line in cell_value.split('\n'):
                    lines += len(line) // 50
                max_lines = max(max_lines, lines)
                
                # 如果文本包含换行符或超过50个字符，设置自动换行 / Set text wrapping if text contains line breaks or exceeds 50 characters
                if '\n' in cell_value or len(cell_value) > 50:
                    cell.alignment = Alignment(wrap_text=True, vertical='top')
        
        # 设置行高（每行大约15个单位，最小15，最大300）/ Set row height (about 15 units per line, min 15, max 300)
        row_height = min(max(15, max_lines * 15), 300)
        worksheet.row_dimensions[row_idx].height = row_height
    
    # 设置标题行的格式 / Set header row format
    for cell in worksheet[1]:
        cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
        worksheet.row_dimensions[1].height = 20
    


if __name__ == "__main__":
    # Simple test
    client = setup_client()
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Normal-sighted parents suddenly gave birth to a color-blind daughter, and the father killed the mother. Why?"}
    ]
    response = get_response(client, messages, model="claude37_sonnet", enable_cot=True)
    print("Main response:")
    print(response["content"])
    print("\nModel's thinking process:")
    print(response["reasoning_content"])