"""Shared tool validation and permission checks for both model protocols."""
from __future__ import annotations

import asyncio
from .tools import check_permission
from .ui import print_tool_call


class ToolExecution:
    async def _run_checked_tool(self, name: str, inp) -> str:
        """Validate and authorize at execution time for both model protocols."""
        from .validation import validate_json
        # 找寻tools中名字匹配的工具(包括mcp+原有工具)
        definition = next((tool for tool in self.tools if tool["name"] == name), None)
        if definition is None:
            return f"Tool not available in this run: {name}"
        # 校验inp跟所需要的input_schema是否匹配
        error = validate_json(inp, definition["input_schema"])
        if error:
            return f"Error: invalid arguments for {name}: {error}"
        # 展示工具调用的名称、图标和输入参数的摘要信息在界面
        print_tool_call(name, inp)
        # 进行权限检查，判断是否允许执行工具调用
        permission = check_permission(name, inp, self.permission_mode, self._plan_file_path)
        if permission["action"] == "deny":
            return f"Action denied: {permission.get('message', '')}"
        if permission["action"] == "confirm" and not await self._confirm_tool_call(name, permission):
            return "User denied this action."
        permission = check_permission(name, inp, self.permission_mode, self._plan_file_path)
        if permission["action"] == "deny":
            return f"Action denied: {permission.get('message', '')}"
        try:
            # 当工具/参数/鉴权都校验完成之后，才开始准备执行tool工具
            return await self._execute_tool_call(name, inp)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return f"Error executing tool: {exc}"

