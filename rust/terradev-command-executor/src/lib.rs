use pyo3::prelude::*;
use pyo3_asyncio::tokio::future_into_py;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::process::Stdio;
use std::sync::Arc;
use tokio::process::Command;
use tokio::sync::Semaphore;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandResult {
    pub success: bool,
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
}

pub struct CommandExecutor {
    max_concurrent: usize,
    semaphore: Arc<Semaphore>,
}

impl CommandExecutor {
    pub fn new(max_concurrent: usize) -> Self {
        Self {
            max_concurrent,
            semaphore: Arc::new(Semaphore::new(max_concurrent)),
        }
    }

    pub async fn execute_command(
        &self,
        program: String,
        args: Vec<String>,
        env: Option<HashMap<String, String>>,
    ) -> CommandResult {
        let _permit = self.semaphore.acquire().await.unwrap();

        let mut cmd = Command::new(&program);
        cmd.args(&args);
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        if let Some(env_vars) = env {
            for (k, v) in env_vars {
                cmd.env(&k, &v);
            }
        }

        match cmd.output().await {
            Ok(output) => {
                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();
                let exit_code = output.status.code().unwrap_or(-1);
                CommandResult {
                    success: output.status.success(),
                    stdout,
                    stderr,
                    exit_code,
                }
            }
            Err(e) => CommandResult {
                success: false,
                stdout: String::new(),
                stderr: e.to_string(),
                exit_code: -1,
            },
        }
    }

    pub async fn execute_parallel(
        &self,
        commands: Vec<(String, Vec<String>, Option<HashMap<String, String>>)>,
    ) -> Vec<CommandResult> {
        let handles = commands
            .into_iter()
            .map(|(program, args, env)| {
                let executor = self.clone();
                tokio::spawn(async move {
                    executor.execute_command(program, args, env).await
                })
            })
            .collect::<Vec<_>>();

        let results = futures::future::join_all(handles)
            .await
            .into_iter()
            .filter_map(|r| r.ok())
            .collect();

        results
    }
}

impl Clone for CommandExecutor {
    fn clone(&self) -> Self {
        Self {
            max_concurrent: self.max_concurrent,
            semaphore: Arc::clone(&self.semaphore),
        }
    }
}

#[pyclass]
pub struct PyCommandExecutor {
    inner: CommandExecutor,
}

#[allow(non_local_definitions)]
#[allow(non_local_definitions)]
#[pymethods]
impl PyCommandExecutor {
    #[new]
    fn new(max_concurrent: usize) -> Self {
        Self {
            inner: CommandExecutor::new(max_concurrent),
        }
    }

    fn execute_command<'p>(
        &self,
        py: Python<'p>,
        program: String,
        args: Vec<String>,
        env: Option<HashMap<String, String>>,
    ) -> PyResult<&'p PyAny> {
        let executor = self.inner.clone();
        future_into_py::<_, PyObject>(py, async move {
            let result = executor.execute_command(program, args, env).await;
            Python::with_gil(|py| {
                let dict = pyo3::types::PyDict::new(py);
                dict.set_item("success", result.success)?;
                dict.set_item("stdout", result.stdout)?;
                dict.set_item("stderr", result.stderr)?;
                dict.set_item("exit_code", result.exit_code)?;
                Ok(dict.into())
            })
        })
    }

    fn execute_parallel<'p>(
        &self,
        py: Python<'p>,
        commands: Vec<(String, Vec<String>, Option<HashMap<String, String>>)>,
    ) -> PyResult<&'p PyAny> {
        let executor = self.inner.clone();
        future_into_py::<_, PyObject>(py, async move {
            let results = executor.execute_parallel(commands).await;
            Python::with_gil(|py| {
                let list = pyo3::types::PyList::empty(py);
                for result in results {
                    let dict = pyo3::types::PyDict::new(py);
                    dict.set_item("success", result.success)?;
                    dict.set_item("stdout", result.stdout)?;
                    dict.set_item("stderr", result.stderr)?;
                    dict.set_item("exit_code", result.exit_code)?;
                    list.append(dict)?;
                }
                Ok(list.into())
            })
        })
    }
}

#[pymodule]
fn terradev_command_executor(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyCommandExecutor>()?;
    Ok(())
}
