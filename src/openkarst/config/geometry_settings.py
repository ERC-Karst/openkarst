#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Geometry backend settings."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class GeometrySettings:
    backend: str = 'circular_analytical'
    table_points: int = 1000
    table_file: Optional[str] = None
    scale_by_diameter: bool = True
