from dataclasses import dataclass
from typing import List, Any, Dict, Optional
from datetime import datetime
import json

@dataclass
class PromptData:
    prompt: str
    key: str
    type: str
    index: int
    resourcePath: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PromptData':
        return cls(
            prompt=data['prompt'],
            key=data['key'],
            type=data['type'],
            index=data['index'],
            resourcePath=data['resourcePath']
        )

@dataclass
class Data:
    promptData: PromptData
    clipId: str
    valid: bool
    lastModifiedTimestamp: str
    filename: str
    sessionId: str
    videoStart: str
    startButtonUpTimestamp: Optional[str] = None
    startButtonDownTimestamp: Optional[str] = None
    swipeForwardTimestamp: Optional[str] = None
    restartButtonDownTimestamp: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Data':
        return cls(
            promptData=PromptData.from_dict(data['promptData']),
            clipId=data['clipId'],
            valid=data['valid'],
            lastModifiedTimestamp=data['lastModifiedTimestamp'],
            filename=data['filename'],
            sessionId=data['sessionId'],
            videoStart=data['videoStart'],
            startButtonUpTimestamp=data.get('startButtonUpTimestamp'),
            startButtonDownTimestamp=data.get('startButtonDownTimestamp'),
            swipeForwardTimestamp=data.get('swipeForwardTimestamp'),
            restartButtonDownTimestamp=data.get('restartButtonDownTimestamp')
        )

    def is_forward(self) -> bool:
        return self.swipeForwardTimestamp is not None

    def is_restart(self) -> bool:
        return self.restartButtonDownTimestamp is not None

    def is_successful(self) -> bool:
        return self.valid and self.is_forward()

    def is_failed_retry(self) -> bool:
        return not self.valid and self.is_restart()

@dataclass
class Full:
    appVersion: str
    key: str
    serverTimestamp: str
    data: Data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Full':
        return cls(
            appVersion=data['appVersion'],
            key=data['key'],
            serverTimestamp=data['serverTimestamp'],
            data=Data.from_dict(data['data'])
        )

@dataclass
class Summary:
    userId: str
    sessionIndex: int
    clipIndex: int
    filename: str
    promptText: str
    valid: bool
    start_s: Optional[float] = None
    end_s: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Summary':
        return cls(
            userId=data['userId'],
            sessionIndex=data['sessionIndex'],
            clipIndex=data['clipIndex'],
            filename=data['filename'],
            promptText=data['promptText'],
            valid=data['valid'],
            start_s=data.get('start_s'),
            end_s=data.get('end_s')
        )

    def has_timing(self) -> bool:
        return self.start_s is not None and self.end_s is not None

    def is_valid_clip(self) -> bool:
        return self.valid and self.has_timing()

    def get_duration(self) -> Optional[float]:
        return (self.end_s - self.start_s) if self.has_timing() else None

    def is_short_clip(self, threshold: float = 2.0) -> bool:
        dur = self.get_duration()
        return dur is not None and dur < threshold

    def is_long_clip(self, threshold: float = 10.0) -> bool:
        dur = self.get_duration()
        return dur is not None and dur > threshold

@dataclass
class Clip:
    summary: Summary
    full: Full

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Clip':
        return cls(
            summary=Summary.from_dict(data['summary']),
            full=Full.from_dict(data['full'])
        )

    def is_successful_attempt(self) -> bool:
        return self.summary.is_valid_clip() and self.full.data.is_successful()

    def is_failed_attempt(self) -> bool:
        return not self.summary.valid and self.full.data.is_failed_retry()

    def is_retry_attempt(self) -> bool:
        return self.full.data.is_restart()

    def is_complete_clip(self) -> bool:
        return self.summary.has_timing() and self.full.data.is_forward()

    def get_prompt_index(self) -> int:
        return self.full.data.promptData.index

    def get_session_id(self) -> str:
        return self.full.data.sessionId

    def get_user_id(self) -> str:
        return self.summary.userId

@dataclass
class MetaData:
    clips: List[Clip]
    sessions: List[Any]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MetaData':
        return cls(
            clips=[Clip.from_dict(c) for c in data.get('clips', [])],
            sessions=data.get('sessions', [])
        )

    @classmethod
    def from_json(cls, json_str: str) -> 'MetaData':
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def from_file(cls, filepath: str) -> 'MetaData':
        with open(filepath, 'r', encoding='utf-8') as f:
            return cls.from_json(f.read())

    def get_valid_clips(self) -> List[Clip]:
        return [c for c in self.clips if c.summary.valid]

    def get_successful_clips(self) -> List[Clip]:
        return [c for c in self.clips if c.is_successful_attempt()]

    def get_failed_clips(self) -> List[Clip]:
        return [c for c in self.clips if c.is_failed_attempt()]

    def get_retry_clips(self) -> List[Clip]:
        return [c for c in self.clips if c.is_retry_attempt()]

    def get_clips_with_timing(self) -> List[Clip]:
        return [c for c in self.clips if c.summary.has_timing()]

    def get_clips_without_timing(self) -> List[Clip]:
        return [c for c in self.clips if not c.summary.has_timing()]

    def get_statistics(self) -> Dict[str, Any]:
        total = len(self.clips)
        valid = len(self.get_valid_clips())
        success = len(self.get_successful_clips())
        retry = len(self.get_retry_clips())
        with_timing = len(self.get_clips_with_timing())
        return {
            'total_clips': total,
            'valid_clips': valid,
            'successful_clips': success,
            'retry_clips': retry,
            'with_timing': with_timing,
            'without_timing': total - with_timing,
            'success_rate': success/total if total else 0,
            'retry_rate': retry/total if total else 0,
        }

def parse_timestamp(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace('Z', '+00:00'))
    except ValueError:
        return None