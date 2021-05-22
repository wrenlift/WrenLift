// Copyright 2021 Zenturi Software Co.
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

pub fn wren_utf8_encode_num_bytes(value: i32) -> i32 {
    assert!(value >= 0, "Cannot encode a negative value.");

    if value <= 0x7f {
        1 as i32
    } 
    else if value <= 0x7ff {
        2 as i32
    }
    else if value <= 0xffff {
        3 as i32
    }
    else if value <= 0x10ffff {
        4 as i32
    }
    else {
        0 as i32
    }
}