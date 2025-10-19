# Problem

> ord('a') -> 97
> chr(97) -> 'a'

## unicode1

a. chr(0) is return '\x00'
b. print it return nothing, the `__repr__()` is '\x00' as what we have seen above
c.

```python
>>> "this is a test" + chr(0) + "string"
'this is a test\x00string'
>>> print("this is a test" + chr(0) + "string")
this is a teststring
```

## unicode 2

a. Using utf-8 instead of utf-16 or utf-32, because utf-8 provides shorter int list.
b. Because the code using `bytes([b]).decode`, which assume that any single bytes can be decodes,
but, '你好' 's encode can't directly decode back for just a single bytes.

```python
>>> '你'.encode('utf-8')
b'\xe4\xbd\xa0'
>>> wrong('你好'.encode('utf-8'))
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<stdin>", line 2, in wrong
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe4 in position 0: unexpected end of data
```

c.

```python
>>> bytes([228]).decode('utf-8')
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe4 in position 0: unexpected end of data
>>> bytes([228, 189]).decode('utf-8')
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
UnicodeDecodeError: 'utf-8' codec can't decode bytes in position 0-1: unexpected end of data
>>> bytes([228, 189, 160]).decode('utf-8')
```

## train_bpe

[train_bpe](cs336_basics/bpe.py)

## train_bpe_tinystories

(a)

Training our bpe model on TinyStories dataset with vocab size 1000, will cost amlost 2 mintues. Reading the result, the longest tokens below, and they are totally make sense.

```shell
Top 5 longest tokens (by bytes):
  1) id=0, len=13 bytes, value=b'<|endoftext|>' (hex=3c7c656e646f66746578747c3e)
  2) id=914, len=11 bytes, value=b' unexpected' (hex=20756e6578706563746564)
  3) id=577, len=10 bytes, value=b' something' (hex=20736f6d657468696e67)
  4) id=896, len=10 bytes, value=b' surprised' (hex=20737572707269736564)
  5) id=995, len=10 bytes, value=b' beautiful' (hex=2062656175746966756c)
```

(b)

```shell
uv run py-spy record -o bpe_profile.svg -- python cs336_basics/bpe.py
```

[`bpe_profile.svg](bpe_profile.svg)

We get the fire graph , and we can see that the most time consuming part is `_apply_merge` function, which is updating token count using increase method.

Before this, the most time comsuming part is file transfer in multiprocessing, which we can optimize by transferring the `start` and `end` index instead of the whole text chunk.

## train_bpe_expts_owt

