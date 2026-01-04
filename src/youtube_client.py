"""
YouTube API Client module.
Handles authentication and fetching comments from YouTube videos.
"""

import re
from typing import Dict, List, Optional, Tuple
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from .config import (
    YOUTUBE_API_SERVICE_NAME,
    YOUTUBE_API_VERSION,
    MAX_COMMENTS_PER_REQUEST,
)


class YouTubeClient:
    """
    Client for interacting with YouTube Data API v3.
    
    Provides methods to fetch video metadata and comments.
    """
    
    # Regex patterns for extracting video ID from various YouTube URL formats
    VIDEO_ID_PATTERNS = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/|youtube\.com\/v\/)([a-zA-Z0-9_-]{11})',
        r'^([a-zA-Z0-9_-]{11})$',  # Direct video ID
    ]
    
    def __init__(self, api_key: str):
        """
        Initialize YouTube client with API key.
        
        Args:
            api_key: YouTube Data API v3 key
        """
        self.api_key = api_key
        self._youtube = None
    
    @property
    def youtube(self):
        """Lazy initialization of YouTube API client."""
        if self._youtube is None:
            self._youtube = build(
                YOUTUBE_API_SERVICE_NAME,
                YOUTUBE_API_VERSION,
                developerKey=self.api_key
            )
        return self._youtube
    
    @staticmethod
    def extract_video_id(url_or_id: str) -> Optional[str]:
        """
        Extract video ID from YouTube URL or validate direct ID.
        
        Args:
            url_or_id: YouTube URL or video ID
            
        Returns:
            Video ID if found, None otherwise
            
        Examples:
            >>> YouTubeClient.extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
            'dQw4w9WgXcQ'
            >>> YouTubeClient.extract_video_id("https://youtu.be/dQw4w9WgXcQ")
            'dQw4w9WgXcQ'
            >>> YouTubeClient.extract_video_id("dQw4w9WgXcQ")
            'dQw4w9WgXcQ'
        """
        url_or_id = url_or_id.strip()
        
        for pattern in YouTubeClient.VIDEO_ID_PATTERNS:
            match = re.search(pattern, url_or_id)
            if match:
                return match.group(1)
        
        return None
    
    def get_video_metadata(self, video_id: str) -> Optional[Dict]:
        """
        Fetch metadata for a YouTube video.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            Dictionary with video metadata or None if not found
        """
        try:
            request = self.youtube.videos().list(
                part="snippet,statistics",
                id=video_id
            )
            response = request.execute()
            
            if not response.get("items"):
                return None
            
            video = response["items"][0]
            snippet = video["snippet"]
            statistics = video.get("statistics", {})
            
            return {
                "video_id": video_id,
                "title": snippet.get("title", ""),
                "channel_title": snippet.get("channelTitle", ""),
                "description": snippet.get("description", ""),
                "published_at": snippet.get("publishedAt", ""),
                "thumbnail_url": snippet.get("thumbnails", {}).get("high", {}).get("url", ""),
                "view_count": int(statistics.get("viewCount", 0)),
                "like_count": int(statistics.get("likeCount", 0)),
                "comment_count": int(statistics.get("commentCount", 0)),
            }
            
        except HttpError as e:
            raise YouTubeAPIError(f"Failed to fetch video metadata: {e}")
    
    def fetch_comments(
        self, 
        video_id: str, 
        max_results: int = 100,
        include_replies: bool = False
    ) -> List[Dict]:
        """
        Fetch comments from a YouTube video.
        
        Args:
            video_id: YouTube video ID
            max_results: Maximum number of comments to fetch
            include_replies: Whether to include reply comments
            
        Returns:
            List of comment dictionaries
        """
        comments = []
        next_page_token = None
        
        try:
            while len(comments) < max_results:
                # Calculate how many comments to request
                remaining = max_results - len(comments)
                request_count = min(remaining, MAX_COMMENTS_PER_REQUEST)
                
                request = self.youtube.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    maxResults=request_count,
                    pageToken=next_page_token,
                    textFormat="plainText",
                    order="time"  # Fetch newest comments first
                )
                response = request.execute()
                
                for item in response.get("items", []):
                    # Get top-level comment
                    top_comment = item["snippet"]["topLevelComment"]["snippet"]
                    comments.append(self._parse_comment(top_comment))
                    
                    # Get replies if requested
                    if include_replies and item["snippet"].get("totalReplyCount", 0) > 0:
                        replies = self._fetch_replies(item["id"])
                        comments.extend(replies)
                    
                    if len(comments) >= max_results:
                        break
                
                next_page_token = response.get("nextPageToken")
                if not next_page_token:
                    break
                    
        except HttpError as e:
            if "commentsDisabled" in str(e):
                raise CommentsDisabledError("Comments are disabled for this video")
            raise YouTubeAPIError(f"Failed to fetch comments: {e}")
        
        return comments[:max_results]
    
    def _fetch_replies(self, parent_id: str, max_replies: int = 10) -> List[Dict]:
        """
        Fetch replies to a comment.
        
        Args:
            parent_id: Parent comment thread ID
            max_replies: Maximum number of replies to fetch
            
        Returns:
            List of reply comment dictionaries
        """
        replies = []
        
        try:
            request = self.youtube.comments().list(
                part="snippet",
                parentId=parent_id,
                maxResults=max_replies,
                textFormat="plainText"
            )
            response = request.execute()
            
            for item in response.get("items", []):
                replies.append(self._parse_comment(item["snippet"], is_reply=True))
                
        except HttpError:
            pass  # Silently ignore reply fetch errors
        
        return replies
    
    @staticmethod
    def _parse_comment(comment_data: Dict, is_reply: bool = False) -> Dict:
        """
        Parse raw comment data into structured format.
        
        Args:
            comment_data: Raw comment snippet from API
            is_reply: Whether this is a reply comment
            
        Returns:
            Structured comment dictionary
        """
        return {
            "text": comment_data.get("textDisplay", ""),
            "author": comment_data.get("authorDisplayName", ""),
            "author_channel_id": comment_data.get("authorChannelId", {}).get("value", ""),
            "published_at": comment_data.get("publishedAt", ""),
            "updated_at": comment_data.get("updatedAt", ""),
            "like_count": comment_data.get("likeCount", 0),
            "is_reply": is_reply,
        }
    
    def validate_api_key(self) -> Tuple[bool, str]:
        """
        Validate that the API key is working.
        
        Returns:
            Tuple of (is_valid, message)
        """
        try:
            # Try to make a simple API call
            self.youtube.videos().list(
                part="snippet",
                id="dQw4w9WgXcQ"  # Known valid video ID
            ).execute()
            return True, "API key is valid"
        except HttpError as e:
            if "forbidden" in str(e).lower():
                return False, "API key is invalid or quota exceeded"
            return False, f"API error: {e}"
        except Exception as e:
            return False, f"Connection error: {e}"


class YouTubeAPIError(Exception):
    """Base exception for YouTube API errors."""
    pass


class CommentsDisabledError(YouTubeAPIError):
    """Exception raised when comments are disabled for a video."""
    pass
