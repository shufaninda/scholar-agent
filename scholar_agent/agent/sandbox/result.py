"""沙箱执行结果数据结构。"""


from pydantic import BaseModel


class SandboxResult(BaseModel):
    """一次沙箱命令执行的统一返回。

    exit_code 约定：0=成功；非 0=命令失败；-1=沙箱层错误（容器不存在/超时/API 错）。
    """

    stdout: str = ""
    stderr: str = ""
    exit_code: int = -1
    error: str = ""
    duration_ms: int = 0

    @property
    def ok(self) -> bool:
        """命令是否成功执行（exit_code == 0 且无沙箱层错误）。"""
        return self.exit_code == 0 and not self.error
